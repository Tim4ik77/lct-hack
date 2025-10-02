# filename: cluster_manager.py
import os, torch, open_clip, zipfile, tempfile
from tqdm import tqdm
from PIL import Image

def add_new_cluster_from_zip(zip_file_bytes, class_name, source_bank_path, output_bank_path, model_config):
    device = model_config["DEVICE"]
    
    print(f"Загрузка модели {model_config['MODEL_NAME']}...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_config['MODEL_NAME'], pretrained=model_config['PRETRAINED'], device=device
    )
    state_dict = torch.load(model_config['CHECKPOINT_PATH'], map_location=device)
    clip_model.visual.load_state_dict(state_dict)
    model = clip_model.visual.eval()

    if not os.path.exists(source_bank_path):
        return {"success": False, "message": f"Исходный файл '{source_bank_path}' не найден."}
        
    original_data = torch.load(source_bank_path)
    if class_name in original_data["class_names"]:
        return {"success": False, "message": f"Класс '{class_name}' уже есть в базе."}

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file_bytes, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        search_path = temp_dir
        content = os.listdir(temp_dir)
        if len(content) == 1 and os.path.isdir(os.path.join(temp_dir, content[0])):
            search_path = os.path.join(temp_dir, content[0])

        image_files = [f for f in os.listdir(search_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            return {"success": False, "message": "В ZIP-архиве не найдено изображений."}

        new_class_embeddings = []
        with torch.no_grad():
            for filename in tqdm(image_files, desc=f"Обработка '{class_name}'"):
                path = os.path.join(search_path, filename)
                try:
                    img = Image.open(path).convert("RGB")
                    img_tensor = preprocess(img).unsqueeze(0).to(device)
                    with torch.amp.autocast(device, enabled=(device == 'cuda')):
                        embedding = model(img_tensor)
                        embedding /= embedding.norm(dim=-1, keepdim=True)
                    new_class_embeddings.append(embedding.cpu())
                except Exception as e: 
                    print(f"Пропуск {filename}: {e}")
        
        if not new_class_embeddings:
            return {"success": False, "message": "Не удалось обработать ни одного изображения."}

        mean_embedding = torch.mean(torch.cat(new_class_embeddings, dim=0), dim=0, keepdim=True)
        mean_embedding /= mean_embedding.norm(dim=-1, keepdim=True)

    original_data["feature_bank"][class_name] = mean_embedding
    original_data["class_names"].append(class_name)

    torch.save(original_data, output_bank_path)
    return {"success": True, "message": f"Банк '{output_bank_path}' успешно обновлен."}