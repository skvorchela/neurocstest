from demoparser2 import DemoParser
import pandas as pd
import os
import glob
import logging
import traceback
from pathlib import Path
from hashlib import md5
from config import RAW_DEMO_FOLDER, PARSED_DATA_PATH, YOUR_STEAMIDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_demo_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1024

def parse_demos() -> pd.DataFrame:
    """Парсинг демо-файлов с улучшенной обработкой ошибок"""
    dfs = []
    demo_paths = [Path(p) for p in glob.glob(str(RAW_DEMO_FOLDER / "*.dem"))]
    
    if not demo_paths:
        logger.error("No .dem files found in %s", RAW_DEMO_FOLDER)
        return pd.DataFrame()

    for path in demo_paths:
        try:
            if not validate_demo_file(path):
                logger.warning("Invalid demo file: %s", path.name)
                continue

            logger.info("Parsing %s...", path.name)
            parser = DemoParser(str(path))
            
            # Базовые свойства, которые гарантированно существуют
            base_properties = [
                "X", "Y", "Z", "health", "is_alive",
                "active_weapon_name", "player_steamid",
                "pitch", "yaw", "FORWARD", "BACK",
                "LEFT", "RIGHT", "FIRE", "ZOOM"
            ]
            
            ticks_data = parser.parse_ticks(base_properties)
            df = pd.DataFrame(ticks_data)
            
            # Генерация уникального match_id на основе хеша файла
            file_hash = md5(path.read_bytes()).hexdigest()[:8]
            df['match_id'] = f"{file_hash}_{len(df)}"
            
            # Фильтрация по SteamID
            df = df[df["player_steamid"].astype(str).isin(YOUR_STEAMIDS)]
            
            df.rename(columns={
                "pitch": "aim_x",
                "yaw": "aim_y",
                "FIRE": "shoot",
                "ZOOM": "aim",
                "FORWARD": "move_forward",
                "BACK": "move_backward",
                "LEFT": "move_left",
                "RIGHT": "move_right"
            }, inplace=True)
            
            dfs.append(df)
            logger.info("Successfully parsed %s", path.name)
            
        except Exception as e:
            logger.error("Error parsing %s: %s\n%s", 
                        path.name, str(e), traceback.format_exc())
            continue

    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        full_df.to_csv(PARSED_DATA_PATH, index=False)
        logger.info("Saved parsed data to %s", PARSED_DATA_PATH)
        return full_df
    
    logger.error("Failed to parse any demo files")
    return pd.DataFrame()

if __name__ == "__main__":
    parse_demos()
    print(parser.available_properties())