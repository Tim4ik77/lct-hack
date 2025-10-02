# filename: inspector.py

import os, cv2, torch, open_clip, numpy as np, re, time
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import easyocr
from datetime import datetime
# Убираем thop, он больше не нужен
# from thop import profile 

class UniversalInstrumentInspector:
    def __init__(self, hf_token): # <-- Убираем yolo_weights_path отсюда
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU'
        
        # Убираем подсчет FLOPs при инициализации
        # self.yolo_flops, self.yolo_params = self.get_yolo_performance(yolo_weights_path)
        
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 20)
            self.sn_font = ImageFont.truetype("DejaVuSans.ttf", 24)
        except IOError:
            print("ПРЕДУПРЕЖДЕНИЕ: Шрифт DejaVuSans.ttf не найден. Кириллица может не отображаться.")
            self.font = ImageFont.load_default()
            self.sn_font = ImageFont.load_default()

        self.MAX_WIDTH = 2048
        self.RMBG_INPUT_SIZE = (1024, 1024)
        self.OUTPUT_DIR = os.path.join('static', 'inspection_results')
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        print(f"Используется устройство: {self.device}")
        
        self.ocr_reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))
        self.bg_remover_model = AutoModelForImageSegmentation.from_pretrained(
            'briaai/RMBG-2.0', token=hf_token, trust_remote_code=True
        ).eval().to(self.device)
        self.bg_remover_transform = transforms.Compose([
            transforms.Resize(self.RMBG_INPUT_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.yolo_model, self.clip_model, self.clip_preprocess, self.feature_bank, self.class_names = [None]*5
        self.similarity_threshold = 0.25

    # --- УДАЛЯЕМ ВСЮ ФУНКЦИЮ get_yolo_performance ---
    # def get_yolo_performance(self, yolo_weights_path):
    #     ...

    def load_models(self, yolo_weights, clip_checkpoint, feature_bank_path, similarity_threshold):
        print(f"Загрузка/перезагрузка моделей с банком: {os.path.basename(feature_bank_path)}")
        self.yolo_model = YOLO(yolo_weights)
        model_name = 'ViT-L-14-quickgelu'
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained='dfn2b', device=self.device
        )
        state_dict = torch.load(clip_checkpoint, map_location=self.device)
        self.clip_model.visual.load_state_dict(state_dict)
        self.clip_model.eval()
        feature_data = torch.load(feature_bank_path, map_location=self.device)
        self.feature_bank = feature_data["feature_bank"]
        self.class_names = feature_data["class_names"]
        self.similarity_threshold = similarity_threshold
        print("--- Модели успешно загружены ---")
        return True

    def _detect_serial_number(self, image_cv):
        try:
            results = self.ocr_reader.readtext(image_cv)
            for (bbox, text, prob) in results:
                if text.replace(" ", "").upper().startswith("AT"): return text
            return None
        except Exception: return None

    def _remove_background(self, pil_image):
        original_size = pil_image.size
        input_tensor = self.bg_remover_transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.bg_remover_model(input_tensor)[-1].sigmoid().cpu()
        pred_pil = transforms.ToPILImage()(preds[0].squeeze())
        mask = pred_pil.resize(original_size, Image.LANCZOS)
        result_pil = pil_image.copy(); result_pil.putalpha(mask); return result_pil

    def inspect_image(self, image_bytes, original_filename, conf_threshold=0.5):
        start_time = time.time()
        if not self.yolo_model: raise Exception("Модели не загружены.")

        np_array = np.frombuffer(image_bytes, np.uint8)
        original_image_cv = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if original_image_cv is None:
            return {"objects": [], "image": np.zeros((100, 100, 3), dtype=np.uint8), "filename": original_filename}

        h, w, _ = original_image_cv.shape
        if w > self.MAX_WIDTH:
            new_w = self.MAX_WIDTH; new_h = int(new_w * (h / w))
            original_image_cv = cv2.resize(original_image_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)

        results = self.yolo_model.predict(source=original_image_cv, conf=conf_threshold, verbose=False)
        detections = results[0].boxes
        
        detected_objects = []
        end_time_pre_process = time.time()

        if len(detections) == 0:
            end_time = time.time()
            return {
                "objects": [], "image": original_image_cv, "filename": original_filename,
                "processing_time": f"{(end_time - start_time):.2f} сек", "gpu_name": self.gpu_name
            }

        debug_image = original_image_cv.copy()
        pil_image_for_drawing = Image.fromarray(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image_for_drawing)

        for i, det in enumerate(detections):
            box = det.xyxy[0].cpu().numpy().astype(int); x1, y1, x2, y2 = box
            cropped_image_cv = original_image_cv[y1:y2, x1:x2]
            if cropped_image_cv.size == 0: continue

            serial_number = self._detect_serial_number(cropped_image_cv)
            
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped_image_cv, cv2.COLOR_BGR2RGB))
            cleaned_pil_rgba = self._remove_background(cropped_pil)
            background_for_clip = Image.new("RGB", cleaned_pil_rgba.size, (0, 0, 0))
            background_for_clip.paste(cleaned_pil_rgba, mask=cleaned_pil_rgba.split()[3])
            
            image_for_clip = self.clip_preprocess(background_for_clip).unsqueeze(0).to(self.device)
            
            with torch.no_grad(), torch.amp.autocast(self.device, enabled=(self.device == 'cuda')):
                query_embedding = self.clip_model.visual(image_for_clip)
                query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

            similarities = {}
            for name, ref in self.feature_bank.items():
                query_float = query_embedding.to(torch.float32)
                ref_float = ref.to(torch.float32)
                similarity = (query_float @ ref_float.T).item()
                similarities[name] = similarity

            best_match_class = max(similarities, key=similarities.get); max_similarity = similarities[best_match_class]
            
            if max_similarity >= self.similarity_threshold:
                predicted_class = best_match_class; color = (0, 255, 0)
            else:
                predicted_class = "Unknown"; color = (0, 165, 255)
            
            detected_objects.append({
                "id": i, "class": predicted_class, "confidence": float(max_similarity),
                "serial_number": serial_number, "box": [int(x1), int(y1), int(x2), int(y2)]
            })
            
            label = f"{predicted_class} ({max_similarity:.2f})"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            text_bbox = draw.textbbox((x1, y1 - 25), label, font=self.font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1 + 2, y1 - 25), label, fill="black", font=self.font)
            
            if serial_number:
                serial_label = f"S/N: {serial_number}"
                sn_text_bbox = draw.textbbox((x1, y2 + 5), serial_label, font=self.sn_font)
                draw.rectangle(sn_text_bbox, fill=(255, 0, 0))
                draw.text((x1 + 2, y2 + 5), serial_label, fill="white", font=self.sn_font)
        
        final_image_cv = cv2.cvtColor(np.array(pil_image_for_drawing), cv2.COLOR_RGB2BGR)
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            "objects": detected_objects, "image": final_image_cv, "filename": original_filename,
            "processing_time": f"{processing_time:.2f} сек", "gpu_name": self.gpu_name
        }