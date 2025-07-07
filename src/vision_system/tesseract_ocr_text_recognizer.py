import cv2
import pytesseract
import time
import threading
import numpy as np
import logging
import os
from langdetect import detect
from .speech_manager import SpeechManager

class TesseractOCRRecognizer:
    def __init__(self, debug_mode=False):
        logging.info("📄 Initializing Enhanced Tesseract OCR text recognition system...")
        self.speech_manager = SpeechManager()
        self.is_processing = False
        self.should_stop = False
        self.last_text = ""
        self.text_cache = {}
        self.cache_expiry = 10
        self.debug_mode = debug_mode

        if self.debug_mode:
            os.makedirs("debug_images", exist_ok=True)

        self.ocr_configs = [
            {'config': '--oem 3 --psm 6', 'name': 'Auto block'},
            {'config': '--oem 3 --psm 7', 'name': 'Single line'},
            {'config': '--oem 3 --psm 8', 'name': 'Single word'},
            {'config': '--oem 3 --psm 13', 'name': 'Raw line'},
            {'config': '--oem 1 --psm 6', 'name': 'LSTM block'},
            {'config': '--oem 3 --psm 3', 'name': 'Auto page'},
        ]

        logging.info("\u2705 Enhanced Tesseract OCR initialized")

    def recognize_text(self, image: np.ndarray, is_capture: bool = False, priority: bool = False):
        if image is None or image.size == 0:
            self.speech_manager.say("Invalid image", priority)
            return None

        if image.shape[0] < 20 or image.shape[1] < 20:
            self.speech_manager.say("Image too small", priority)
            return None

        if is_capture:
            self.speech_manager.say("Processing image for text", priority)

        self.should_stop = False
        self.is_processing = True

        try:
            if self.debug_mode:
                cv2.imwrite("debug_images/original.jpg", image)

            text = self._try_multiple_strategies(image)

            if not text:
                self.speech_manager.say("No readable text", priority)
                return None

            self.last_text = text
            self._cache_text(text)

            if is_capture:
                self.speech_manager.say("Text successfully detected", priority)

            self._speak_text(text, priority)
            return text

        except Exception as e:
            logging.error(f"\u274c Enhanced OCR error: {str(e)}")
            self.speech_manager.say("Error processing text", priority)
            return None
        finally:
            self.is_processing = False
            self.should_stop = False

    def _try_multiple_strategies(self, image):
        strategies = [
            {'name': 'Enhanced', 'func': self._preprocess_enhanced},
            {'name': 'High Contrast', 'func': self._preprocess_high_contrast},
            {'name': 'Morphological', 'func': self._preprocess_morphological},
            {'name': 'Simple', 'func': self._preprocess_simple},
            {'name': 'Denoised', 'func': self._preprocess_denoised},
        ]

        best_text = ""
        best_confidence = 0

        for strategy in strategies:
            try:
                processed_image = strategy['func'](image)

                if self.debug_mode:
                    cv2.imwrite(f"debug_images/processed_{strategy['name'].lower().replace(' ', '_')}.jpg", processed_image)

                for config in self.ocr_configs:
                    try:
                        text = pytesseract.image_to_string(processed_image, config=config['config']).strip()

                        if text:
                            confidence = self._calculate_confidence(text, processed_image)

                            if self.debug_mode:
                                logging.info(f"Strategy: {strategy['name']}, Config: {config['name']}, "
                                             f"Confidence: {confidence:.2f}, Text: '{text[:50]}...'")

                            if confidence > best_confidence and len(text) > 2:
                                best_text = text
                                best_confidence = confidence

                                if confidence > 0.8:
                                    cleaned_text = self._clean_text(best_text)
                                    if cleaned_text:
                                        return cleaned_text

                    except Exception as e:
                        if self.debug_mode:
                            logging.warning(f"OCR config {config['name']} failed: {e}")
                        continue

            except Exception as e:
                if self.debug_mode:
                    logging.warning(f"Preprocessing strategy {strategy['name']} failed: {e}")
                continue

        if best_text:
            cleaned_text = self._clean_text(best_text)
            if cleaned_text:
                return cleaned_text

        return None

    def _preprocess_enhanced(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        height, width = gray.shape
        if width < 1000:
            scale = min(2.0, 1000 / width)
            gray = cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)

        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    def _preprocess_high_contrast(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _preprocess_morphological(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 15, 8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    def _preprocess_simple(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        height, width = gray.shape
        if width < 800:
            scale = 800 / width
            gray = cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)

    def _preprocess_denoised(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _calculate_confidence(self, text, processed_image):
        if not text or len(text.strip()) < 2:
            return 0.0

        confidence = 0.0
        length_factor = min(len(text.strip()) / 20.0, 1.0)
        confidence += length_factor * 0.3
        diversity_factor = min(len(set(text.lower())) / 10.0, 1.0)
        confidence += diversity_factor * 0.3
        if len(text.split()) > 1:
            confidence += 0.2
        common_patterns = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
        pattern_factor = min(sum(1 for p in common_patterns if p in text.lower()) / 3.0, 1.0)
        confidence += pattern_factor * 0.2

        return min(confidence, 1.0)

    def _clean_text(self, text):
        lines = [line.strip().replace('|', 'I').replace('0', 'O').replace('5', 'S')
                 for line in text.splitlines() if len(line.strip()) > 1]
        return " ".join(" ".join(lines).split()) if lines else ""

    def _speak_text(self, text, priority):
        if not text:
            return

        cache_key = self._get_cache_key(text)
        now = time.time()
        self.text_cache = {k: v for k, v in self.text_cache.items() if now - v < self.cache_expiry}

        if cache_key in self.text_cache:
            self.speech_manager.say("Same text detected", priority)
            return

        self.speech_manager.say("Reading text", priority)
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        chunks = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                if len(sentence) > 100:
                    chunks.extend(part.strip() for part in sentence.split(',') if part.strip())
                else:
                    chunks.append(sentence)

        if not chunks:
            words = text.split()
            chunk, chunks = [], []
            for word in words:
                chunk.append(word)
                if len(" ".join(chunk)) > 50:
                    chunks.append(" ".join(chunk))
                    chunk = []
            if chunk:
                chunks.append(" ".join(chunk))

        for i, chunk in enumerate(chunks):
            if self.should_stop:
                self.speech_manager.say("Reading interrupted", priority)
                break
            if chunk.strip():
                self.speech_manager.say(chunk.strip(), priority)
                if i < len(chunks) - 1:
                    time.sleep(0.3)

    def get_debug_info(self):
        if not self.debug_mode:
            return "Debug mode not enabled"
        files = os.listdir("debug_images") if os.path.exists("debug_images") else []
        return f"Debug images saved: {len(files)}\n" + "\n".join(f"  - {f}" for f in files if f.endswith('.jpg'))

    def enable_debug(self):
        self.debug_mode = True
        os.makedirs("debug_images", exist_ok=True)
        logging.info("Debug mode enabled - images will be saved to debug_images/")

    def disable_debug(self):
        self.debug_mode = False
        logging.info("Debug mode disabled")

    def _get_cache_key(self, text):
        return "".join(text.lower().split())[:50]

    def _cache_text(self, text):
        self.text_cache[self._get_cache_key(text)] = time.time()

    def interrupt(self):
        self.should_stop = True
        self.speech_manager.interrupt()

    def cleanup(self):
        self.should_stop = True
        if hasattr(self, 'speech_manager'):
            self.speech_manager.cleanup()
