import cv2
import pytesseract
import numpy as np
import os
import sys

class TextRecognizer:
    def __init__(self):
        """Initialize OCR system."""
        if sys.platform.startswith('win'):
            # Check common installation paths on Windows
            tesseract_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            ]
            
            for path in tesseract_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
            else:
                print("\nWARNING: Tesseract not found in common locations!")
                print("Please ensure Tesseract is installed and the path is correct.")
                print("Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
                print("Default path should be: C:\\Program Files\\Tesseract-OCR\\tesseract.exe")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for better text extraction
        # This works better with varying lighting conditions
        adaptive_threshold = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to remove noise
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_OPEN, kernel)
        
        # Apply dilation to make text thicker and more readable
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(opening, kernel, iterations=1)
        
        # Apply bilateral filter to smooth the image while preserving edges
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Create multiple preprocessed versions for testing
        _, otsu = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Return the best version based on the image type
        # For high contrast text like in the example, otsu often works best
        return otsu

    def recognize_text(self, image: np.ndarray) -> str:
        """Perform OCR on the image."""
        try:
            # Create multiple preprocessed versions
            preprocessed1 = self.preprocess_image(image)
            
            # Try different preprocessing techniques
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, preprocessed2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Try with different Tesseract configurations
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ\'\s"'
            text1 = pytesseract.image_to_string(preprocessed1, config=custom_config)
            
            # Try with different PSM mode for single text block
            custom_config2 = r'--oem 3 --psm 4'
            text2 = pytesseract.image_to_string(preprocessed2, config=custom_config2)
            
            # Try with original image as well
            text3 = pytesseract.image_to_string(image, config=r'--oem 3 --psm 11')
            
            # Combine results, prioritizing the one with more content
            texts = [text1, text2, text3]
            best_text = max(texts, key=len).strip()
            
            # Save all preprocessed images for debugging
            self._save_debug_images(image, preprocessed1, preprocessed2)
            
            return best_text
        except pytesseract.TesseractNotFoundError:
            print("\nERROR: Tesseract is not installed or not found in PATH")
            print("Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("Make sure to add it to PATH during installation")
            return ""
        except Exception as e:
            print(f"\nError performing OCR: {str(e)}")
            return ""
    
    def _save_debug_images(self, original, *preprocessed_images):
        """Save debug images to help diagnose OCR issues."""
        debug_dir = "ocr_debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
            
        timestamp = cv2.getTickCount()
        cv2.imwrite(f"{debug_dir}/original_{timestamp}.jpg", original)
        
        for i, img in enumerate(preprocessed_images):
            cv2.imwrite(f"{debug_dir}/preprocessed_{i}_{timestamp}.jpg", img)