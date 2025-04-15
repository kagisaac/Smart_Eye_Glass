import cv2
import numpy as np
import pytesseract
import pyttsx3
import threading
import queue
from typing import Optional
from langdetect import detect
import time
from .speech_manager import SpeechManager

class TextRecognizer:
    def __init__(self):
        """Initialize text recognition with optimized settings."""
        # OCR Configuration
        self.custom_config = r'--oem 3 --psm 6'
        
        # TTS Queue and Thread
        self.tts_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.is_running = True
        self.tts_thread.start()
        
        # Track last spoken text and time
        self.last_text = ""
        self.last_spoken_time = 0
        self.min_speak_interval = 1.0  # seconds
        
        # Initialize speech manager for system announcements
        self.speech_manager = SpeechManager()
        
        print("✅ Text Recognizer initialized")

    def recognize_text(self, image: np.ndarray) -> Optional[str]:
        """Recognize text in image."""
        try:
            # Skip if image is too small
            if image.shape[0] < 20 or image.shape[1] < 20:
                return None

            # Announce start of recognition
            self.speech_manager.say("Starting text recognition")

            # Preprocess image
            processed = self._preprocess_image(image)
            
            # Perform OCR
            text = pytesseract.image_to_string(processed, config=self.custom_config).strip()
            
            if not text:
                self.speech_manager.say("No text detected")
                return None

            current_time = time.time()
            
            # Only speak if the text is different or enough time has passed
            if (text != self.last_text or 
                current_time - self.last_spoken_time >= self.min_speak_interval):
                self.last_text = text
                self.last_spoken_time = current_time
                self._queue_text_for_speaking(text)
            
            return text

        except Exception as e:
            print(f"❌ Error in text recognition: {str(e)}")
            self.speech_manager.say("Error during text recognition")
            return None

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Optimize image for OCR."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(binary)
            
            # Enhance contrast
            enhanced = cv2.convertScaleAbs(denoised, alpha=1.5, beta=0)
            
            return enhanced
            
        except Exception as e:
            print(f"❌ Error in image preprocessing: {str(e)}")
            return image

    def _queue_text_for_speaking(self, text: str):
        """Queue text for TTS processing."""
        try:
            # Detect language
            lang = detect(text)
            self.tts_queue.put((text, lang))
        except Exception as e:
            print(f"❌ Error queueing text: {str(e)}")

    def _tts_worker(self):
        """Background worker for TTS processing."""
        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            
            while self.is_running:
                try:
                    # Get text from queue with timeout
                    text, lang = self.tts_queue.get(timeout=0.5)
                    
                    # Set voice based on language
                    voices = engine.getProperty('voices')
                    for voice in voices:
                        if lang in voice.languages:
                            engine.setProperty('voice', voice.id)
                            break
                    
                    # Speak text
                    engine.say(text)
                    engine.runAndWait()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ Error in TTS: {str(e)}")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"❌ Error initializing TTS engine: {str(e)}")
        finally:
            if engine:
                engine.stop()

    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        if self.tts_thread.is_alive():
            self.tts_thread.join(timeout=1.0)