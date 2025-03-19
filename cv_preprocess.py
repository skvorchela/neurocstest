import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Класс для расширенной предобработки данных"""
    def __init__(self):
        self.angle_scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def normalize_angles(self, df):
        """Нормализация углов прицеливания"""
        df[['aim_x', 'aim_y']] = self.angle_scaler.fit_transform(df[['aim_x', 'aim_y']])
        return df

def load_and_preprocess_data(test_size=0.2):
    """Улучшенный пайплайн предобработки"""
    logger.info("Starting advanced preprocessing...")
    
    # 1. Загрузка данных
    df = pd.read_csv(PARSED_DATA_PATH)
    
    # 2. Инжиниринг признаков
    engineer = FeatureEngineer()
    df = engineer.normalize_angles(df)
    
    # 3. Препроцессинг
    numeric_features = ["X", "Y", "Z", "health"]
    categorical_features = ["active_weapon_name"]
    
    preprocessor = ColumnTransformer([
        ("num", MinMaxScaler(), numeric_features),
        ("cat", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            max_categories=25,
            min_frequency=50
        ), categorical_features)
    ])
    
    # 4. Применение преобразований
    X_processed = preprocessor.fit_transform(df)
    
    # 5. Обновление конфига
    MODEL_CONFIG['input_size'] = X_processed.shape[1]
    logger.info(f"Updated input size: {MODEL_CONFIG['input_size']}")
    
    # 6. Разделение данных
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df['match_id']))
    
    # 7. Подготовка целевых переменных
    action_cols = ["move_forward", "move_backward", "move_left", "move_right", "shoot"]
    aim_cols = ["aim_x", "aim_y"]
    
    # 8. Сохранение артефактов
    joblib.dump(preprocessor, PROCESSED_DATA_FOLDER / "preprocessor.joblib")
    
    return (
        X_processed[train_idx], 
        X_processed[test_idx],
        df.iloc[train_idx][action_cols].values,
        df.iloc[test_idx][action_cols].values,
        df.iloc[train_idx][aim_cols].values,
        df.iloc[test_idx][aim_cols].values
    )

if __name__ == "__main__":
    load_and_preprocess_data()