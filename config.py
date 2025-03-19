import os
from pathlib import Path
import yaml

# Правильное определение базовой директории
BASE_DIR = Path(__file__).parent  # Теперь указывает на папку с проектом
CONFIG_PATH = BASE_DIR / 'config.yml'  # Путь к config.yml в папке проекта

# Проверка существования файла
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Файл конфигурации {CONFIG_PATH} не найден!")

# Загрузка конфига YAML
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# Пути
RAW_DEMO_FOLDER = Path(config['paths']['raw_demos'])
PARSED_DATA_PATH = Path(config['paths']['parsed_data'])
PROCESSED_DATA_FOLDER = Path(config['paths']['processed_data'])
MODEL_SAVE_PATH = Path(config['paths']['model'])
YOLO_MODEL_PATH = Path(config['paths']['yolo_model'])
LOGS_DIR = Path(config['paths']['logs'])

# Создание директорий
for p in [RAW_DEMO_FOLDER, PARSED_DATA_PATH.parent, PROCESSED_DATA_FOLDER, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Настройки CV
SCREEN_REGION = tuple(config['cv']['screen_region'])
YOLO_CONF = config['cv']['yolo_conf']

# Настройки модели
MODEL_CONFIG = config['model']

# Параметры RL
RL_CONFIG = config['rl']

# SteamID и разрешения экрана
YOUR_STEAMIDS = config['steam_ids']
RESOLUTIONS = config['resolutions']

# Настройки OCR
OCR_CONFIG = config['ocr']