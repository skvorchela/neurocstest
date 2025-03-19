import time
import numpy as np
import tensorflow as tf
import logging
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyController
from vision import GameVision
from config import SCREEN_REGION, MODEL_SAVE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CS2Bot:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.vision = GameVision()
        self.mouse = MouseController()
        self.keyboard = KeyController()
        self.current_keys = {
            'w': False, 's': False,
            'a': False, 'd': False
        }
        self.running = False
        
    def _update_keys(self, actions):
        """Обновление состояния клавиш движения"""
        move_actions = {
            'w': actions[0] > 0.7,
            's': actions[1] > 0.7,
            'a': actions[2] > 0.7,
            'd': actions[3] > 0.7
        }
        
        for key in move_actions:
            if move_actions[key] != self.current_keys[key]:
                if move_actions[key]:
                    self.keyboard.press(key)
                else:
                    self.keyboard.release(key)
                self.current_keys[key] = move_actions[key]

    def _apply_aim(self, aim_vector):
        """Преобразование предсказаний в движение мыши"""
        sensitivity = 0.5  # Настройте под свою чувствительность в игре
        dx = aim_vector[0] * sensitivity * SCREEN_REGION[2]
        dy = aim_vector[1] * sensitivity * SCREEN_REGION[3]
        self.mouse.move(int(dx), int(dy))
        
        # Обработка стрельбы
        if actions[4] > 0.5:
            self.mouse.click(Button.left)

    def _get_game_state(self):
        """Получение и нормализация состояния игры"""
        state = {
            'health': self.vision.read_health(),
            'armor': self.vision.read_armor(),
            'players': self.vision.detect_objects()['players'],
            'weapons': self.vision.detect_objects()['weapons']
        }
        
        # Нормализация
        return np.array([
            state['health'] / 100,
            state['armor'] / 100,
            len(state['players'])/5,
            len(state['weapons'])/10
        ])

    def run(self, duration=60):
        """Запуск бота на указанное время в секундах"""
        logger.info("Starting CS2 Bot...")
        self.running = True
        start_time = time.time()
        
        try:
            while self.running and (time.time() - start_time) < duration:
                # Получение состояния
                state = self._get_game_state()
                
                # Предсказание действий
                actions, aim = self.model.predict(
                    np.expand_dims(state, 0), 
                    verbose=0
                )
                
                # Применение действий
                self._update_keys(actions[0])
                self._apply_aim(aim[0])
                
                # Задержка между тиками
                time.sleep(0.033)  # ~30 FPS

        except Exception as e:
            logger.error(f"Bot error: {str(e)}")
        finally:
            self.stop()
            
    def stop(self):
        """Безопасная остановка бота"""
        self.running = False
        for key in self.current_keys:
            if self.current_keys[key]:
                self.keyboard.release(key)
        self.mouse.release(Button.left)
        logger.info("Bot stopped safely")

if __name__ == "__main__":
    bot = CS2Bot(MODEL_SAVE_PATH)
    
    # Запуск на 5 минут (300 секунд)
    bot.run(300)