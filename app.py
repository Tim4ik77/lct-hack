# filename: app.py
import streamlit as st
import os, io, zipfile, cv2, torch, tempfile, json, re
from datetime import datetime
from inspector import UniversalInstrumentInspector
from cluster_manager import add_new_cluster_from_zip

# --- КОНФИГУРАЦИЯ ---
YOLO_WEIGHTS_PATH = 'models/best_medium.pt'
CLIP_CHECKPOINT_PATH = 'models/best_encoder_3class.pth'
FEATURE_BANK_DIR = 'feature_banks'
LOG_FILE = "detection_log.jsonl"
HF_TOKEN = ""

FULL_TOOL_SET = {
    '10_вырезанный_ключ_рожковый', 
    '11_вырезанные_бокорезы', 
    '4_вырезанный_коловорот', 
    '5_вырезанные_пассатижи_контровочные', 
    '6_вырезанные_пассатижи', 
    '7_вырезанная_шэрница', 
    '8_вырезанный_разводной_ключ', 
    '9_вырезанная_открывашка', 
    '1_вырезанная_отвертка «-»', 
    '3_вырезанная_отвертка_на_смещённый_крест', 
    '2_вырезанная_отвертка «+»'
}

def clear_folder(folder_path):
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    for filename in os.listdir(folder_path):
        try: os.unlink(os.path.join(folder_path, filename))
        except Exception as e: print(f'Failed to delete {filename}. Reason: {e}')

@st.cache_resource
def load_inspector():
    RESULTS_DIR = os.path.join('static', 'inspection_results')
    clear_folder(RESULTS_DIR)
    # Убираем передачу yolo_weights_path, она больше не нужна в __init__
    inspector_instance = UniversalInstrumentInspector(hf_token=HF_TOKEN)
    return inspector_instance

inspector = load_inspector()
if 'last_results' not in st.session_state: st.session_state.last_results = {}

def log_detection(result):
    log_entry = {"timestamp": datetime.now().isoformat(), "filename": result["filename"], "objects": result["objects"]}
    with open(LOG_FILE, "a", encoding="utf-8") as f: f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def get_last_n_logs(n):
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f: lines = f.readlines()
        return [json.loads(line) for line in lines[-n:]]
    except FileNotFoundError: return []

def create_audit_report(logs):
    report = {}
    for log in logs:
        filename_key = re.sub(r'\s*\(\d+\)$', '', log['filename'])
        detected_classes = {obj['class'] for obj in log['objects'] if obj['class'] != 'Unknown'}
        missing_tools = list(FULL_TOOL_SET - detected_classes)
        report[filename_key] = {"all_present": len(missing_tools) == 0, "missing_tools": sorted(missing_tools)}
    return report

st.set_page_config(layout="wide", page_title="Instrument Inspector AI")
st.markdown("""
<style>
section[data-testid="stSidebar"] {
        width: 425px !important;
        max-width: 425px !important;
}
</style>""", unsafe_allow_html=True) # Ваш CSS без изменений

with st.sidebar:
    st.image("lct_logo.png", width=300)
    st.title("Instrument Inspector")
    st.header("Настройки анализатора")
    
    feature_bank_files = [f for f in os.listdir(FEATURE_BANK_DIR) if f.endswith('.pt')]
    if not feature_bank_files: st.error(f"Папка '{FEATURE_BANK_DIR}' пуста!"); st.stop()
        
    selected_bank = st.selectbox("Активный банк признаков:", feature_bank_files, key='selected_bank')
    
    if st.button("Загрузить/Обновить модели"):
        bank_path = os.path.join(FEATURE_BANK_DIR, selected_bank)
        with st.spinner('Загрузка AI моделей...'):
            inspector.load_models(YOLO_WEIGHTS_PATH, CLIP_CHECKPOINT_PATH, bank_path, 0.25)
        st.success(f"Модели успешно загружены!")
        st.session_state.last_results = {}

    st.markdown("---")
    st.header("Информация о системе")
    st.metric(label="Устройство", value=inspector.gpu_name)
    # --- УДАЛЯЕМ ОТОБРАЖЕНИЕ FLOPs и ПАРАМЕТРОВ ---
    # st.metric(label="YOLOv8-M FLOPs", value=inspector.yolo_flops)
    # st.metric(label="YOLOv8-M Параметры", value=inspector.yolo_params)
    st.caption("CLIP Модель: ViT-L-14")
    
    st.markdown("---")
    st.header("История и отчеты")
    num_last_detections = st.number_input("Количество последних детекций:", min_value=1, value=10)
    
    logs_data = get_last_n_logs(num_last_detections)
    st.download_button("Скачать лог детекций (.json)", json.dumps(logs_data, indent=2, ensure_ascii=False), f"last_{num_last_detections}_detections.json", "application/json")
    
    report_data = create_audit_report(logs_data)
    st.download_button("Скачать отчет о комплектности (.json)", json.dumps(report_data, indent=2, ensure_ascii=False), f"audit_report_last_{num_last_detections}.json", "application/json")

st.header("Панель управления")
tab1, tab2 = st.tabs(["🔍 Инспекция изображений", "➕ Добавить новый кластер"])

with tab1:
    st.subheader("Загрузите изображение или ZIP-архив")
    uploaded_files = st.file_uploader("Выберите файлы...", type=['png', 'jpg', 'jpeg', 'zip'], accept_multiple_files=True, key="uploader")

    if uploaded_files:
        current_file_ids = {f"{f.name}-{f.size}" for f in uploaded_files}
        if st.session_state.get('processed_files') != current_file_ids:
            st.session_state.last_results = {}
            st.session_state.processed_files = current_file_ids

        if inspector.yolo_model is None:
            st.warning("Модели не загружены. Нажмите 'Загрузить/Обновить модели' в боковой панели.")
        else:
            # --- ИЗМЕНЕНИЕ: Обрабатываем список файлов в ОБРАТНОМ ПОРЯДКЕ ---
            for uploaded_file in uploaded_files[::-1]:
                file_id = f"{uploaded_file.name}-{uploaded_file.size}"
                if file_id not in st.session_state.last_results:
                    with st.spinner(f"Анализ `{uploaded_file.name}`..."):
                        if uploaded_file.type == "application/zip":
                            zip_results = []
                            with tempfile.TemporaryDirectory() as temp_dir:
                                with zipfile.ZipFile(uploaded_file, 'r') as zf: zf.extractall(temp_dir)
                                for root, _, files in os.walk(temp_dir):
                                    for filename in files:
                                        if filename.lower().endswith(('.png', '.jpg', 'jpeg')):
                                            with open(os.path.join(root, filename), 'rb') as f: image_bytes = f.read()
                                            result = inspector.inspect_image(image_bytes, original_filename=filename)
                                            log_detection(result); zip_results.append(result)
                            st.session_state.last_results[file_id] = zip_results
                        else:
                            image_bytes = uploaded_file.getvalue()
                            result = inspector.inspect_image(image_bytes, original_filename=uploaded_file.name)
                            log_detection(result); st.session_state.last_results[file_id] = [result]
                
                results_to_display = st.session_state.last_results[file_id]
                st.markdown("---"); st.subheader(f"Результаты для файла: `{uploaded_file.name}`")
                if not results_to_display: st.warning("В ZIP-архиве не найдено изображений.")
                
                for result in results_to_display:
                    st.text(f"Файл: {result['filename']}")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.image(cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB), caption="Результат детекции")
                    with col2:
                        st.info(f"**Время анализа:** {result['processing_time']} (на {result['gpu_name']})")
                        st.json(result['objects'])
                        st.download_button("Скачать JSON", json.dumps(result['objects'], indent=2, ensure_ascii=False), f"{os.path.splitext(result['filename'])[0]}_detection.json", "application/json", key=f"dl_{file_id}_{result['filename']}")

with tab2:
    st.subheader("Создание нового банка с дополнительным классом")
    with st.form("add_cluster_form"):
        new_cluster_zip = st.file_uploader("ZIP-архив с изображениями", type=['zip'])
        new_class_name = st.text_input("Имя нового класса")
        source_bank = st.selectbox("Добавить в копию банка:", feature_bank_files)
        submitted = st.form_submit_button("Создать новый банк")
        if submitted:
            if not all([new_cluster_zip, new_class_name, source_bank]): st.error("Заполните все поля.")
            else:
                with st.spinner(f"Добавляем класс '{new_class_name}'..."):
                    source_bank_path = os.path.join(FEATURE_BANK_DIR, source_bank)
                    output_bank_path = os.path.join(FEATURE_BANK_DIR, f"{os.path.splitext(source_bank)[0]}_plus.pt")
                    zip_bytes = io.BytesIO(new_cluster_zip.getvalue())
                    model_config = {"CHECKPOINT_PATH": CLIP_CHECKPOINT_PATH, "MODEL_NAME": 'ViT-L-14-quickgelu', "PRETRAINED": 'dfn2b', "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"}
                    result = add_new_cluster_from_zip(zip_bytes, new_class_name, source_bank_path, output_bank_path, model_config)
                    if result['success']:
                        st.success(result['message']); st.info("Страница будет перезагружена."); st.session_state.last_results = {}; st.rerun()
                    else: st.error(result['message'])