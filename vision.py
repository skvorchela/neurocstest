import cv2
import pytesseract
import numpy as np
import mss
from ultralytics import YOLO
from config import *

# В vision.py добавьте:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class GameVision:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)
        self.sct = mss.mss()
        
    def _preprocess_roi(self, img):
        """Подготовка изображения для OCR"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        return cv2.GaussianBlur(thresh, (3,3), 0)

    def read_health(self):
        img = np.array(self.sct.grab({
            'left': HEALTH_REGION[0], 'top': HEALTH_REGION[1],
            'width': HEALTH_REGION[2]-HEALTH_REGION[0],
            'height': HEALTH_REGION[3]-HEALTH_REGION[1]}))
        processed = self._preprocess_roi(img)
        text = pytesseract.image_to_string(processed, config=OCR_CONFIG)
        return int(text) if text.strip().isdigit() else 100

    def read_armor(self):
        img = np.array(self.sct.grab({
            'left': ARMOR_REGION[0], 'top': ARMOR_REGION[1],
            'width': ARMOR_REGION[2]-ARMOR_REGION[0],
            'height': ARMOR_REGION[3]-ARMOR_REGION[1]}))
        processed = self._preprocess_roi(img)
        text = pytesseract.image_to_string(processed, config=OCR_CONFIG)
        return int(text) if text.strip().isdigit() else 0

    def detect_objects(self):
        img = np.array(self.sct.grab(SCREEN_REGION))[:, :, :3]
        results = self.model.predict(
            source=img,
            conf=YOLO_CONF, 
            classes=[0, 1, 4],  # Игроки, бомба, оружие
            imgsz=1280,
            verbose=False
        )
        return {
            'players': results[0].boxes.xyxy.cpu().numpy(),
            'bomb': results[0].boxes.cls.cpu().numpy() == 1,
            'weapons': results[0].boxes.cls.cpu().numpy() == 4
        }