import cv2
import numpy as np
import easyocr
import pyttsx3
import time
from typing import Optional

class TextRecognizer:
    def __init__(self):
        """Initialize text recognition with EasyOCR and text-to-speech."""
        print("Initializing EasyOCR (this may take a moment)...")
        self.reader = easyocr.Reader(['en'], gpu=False)

        # Speech setup
        self.speech_manager = pyttsx3.init()
        self.speech_manager.setProperty('rate', 150)
        self.speech_manager.setProperty('volume', 1.0)

        # State flags
        self.should_stop = False
        self.is_processing = False

        print("✅ Text Recognizer initialized")

    def recognize_text(self, image: np.ndarray, is_capture: bool = False) -> Optional[str]:
        """Recognize and optionally read text from the given image."""
        if self.is_processing:
            self.speak("Already processing text. Please wait.")
            return None

        self.is_processing = True
        self.should_stop = False
        best_text = ""
        highest_confidence = 0.0

        try:
            if image.shape[0] < 20 or image.shape[1] < 20:
                self.speak("Image is too small for text recognition.")
                return None

            if is_capture:
                self.speak("Processing captured image for text recognition.")

            # Generate preprocessed images
            preprocessed_images = self._get_preprocessed_versions(image)

            for processed_img in preprocessed_images:
                if self.should_stop:
                    self.speak("Text recognition interrupted.")
                    return None

                results = self.reader.readtext(processed_img)
                if results:
                    texts = [text for _, text, _ in results]
                    confidences = [conf for _, _, conf in results]
                    avg_conf = sum(confidences) / len(confidences)

                    combined_text = ' '.join(texts)
                    if avg_conf > highest_confidence and self._is_valid_text(combined_text):
                        best_text = combined_text
                        highest_confidence = avg_conf

            if not best_text:
                self.speak("No readable text detected in the image.")
                return None

            # Split and read detected text
            sentences = self._split_into_sentences(best_text)
            self.speak("Starting to read the detected text.")
            for sentence in sentences:
                if self.should_stop:
                    self.speak("Text reading interrupted.")
                    break
                self.speak(sentence)
                time.sleep(0.5)

            if not self.should_stop:
                self.speak("Finished reading the text.")

            return best_text

        except Exception as e:
            print(f"❌ Error during text recognition: {e}")
            self.speak("An error occurred during text recognition.")
            return None

        finally:
            self.is_processing = False
            self.should_stop = False

    def interrupt(self):
        """Interrupt the recognition and speech process."""
        self.should_stop = True
        print("🛑 Interrupt signal received.")

    def _get_preprocessed_versions(self, image: np.ndarray) -> list:
        """Generate various enhanced versions of the image for better OCR."""
        versions = [image]

        try:
            # Grayscale version
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                versions.append(gray)

            # CLAHE-enhanced version
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced_lab = cv2.merge((cl, a, b))
                enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                versions.append(enhanced_bgr)

            # Denoised version
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            versions.append(denoised)

        except Exception as e:
            print(f"⚠️ Preprocessing error: {e}")

        return versions

    def _is_valid_text(self, text: str) -> bool:
        """Ensure recognized text is meaningful (has enough alphabetic content)."""
        if not text:
            return False
        letter_ratio = sum(c.isalpha() for c in text) / max(1, len(text))
        return letter_ratio > 0.2

    def _split_into_sentences(self, text: str) -> list:
        """Split recognized text into readable sentences."""
        sentences = []
        sentence = ""
        for char in text:
            sentence += char
            if char in '.!?':
                if sentence.strip():
                    sentences.append(sentence.strip())
                    sentence = ""
        if sentence.strip():
            sentences.append(sentence.strip())
        return sentences

    def speak(self, text: str):
        """Speak the provided text using TTS."""
        try:
            if not self.should_stop:
                print(f"🔊 Speaking: {text}")
                self.speech_manager.say(text)
                self.speech_manager.runAndWait()
        except Exception as e:
            print(f"❌ Speech error: {e}")

    def cleanup(self):
        """Stop ongoing speech and mark for interruption."""
        self.should_stop = True
        if hasattr(self, 'speech_manager'):
            self.speech_manager.stop()
