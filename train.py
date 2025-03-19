import numpy as np
import tensorflow as tf
from parse import parse_demos
from preprocess import load_and_preprocess_data
from model import build_enhanced_model, get_callbacks
from config import MODEL_CONFIG, MODEL_SAVE_PATH, LOGS_DIR  # Добавили импорт

def train_supervised():
    # Парсинг и предобработка данных
    parse_demos()
    X_train, X_test, y_act_train, y_act_test, y_aim_train, y_aim_test = load_and_preprocess_data()
    
    # Построение модели
    model = build_enhanced_model()
    
    # Обучение
    history = model.fit(
        x=X_train,
        y={'actions': y_act_train, 'aim': y_aim_train},
        validation_data=(X_test, {'actions': y_act_test, 'aim': y_aim_test}),
        epochs=MODEL_CONFIG['epochs'],
        batch_size=MODEL_CONFIG['batch_size'],
        callbacks=get_callbacks(),
        verbose=2
    )
    
    # Сохранение
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    train_supervised()