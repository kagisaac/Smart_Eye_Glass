import cv2
import numpy as np
import pytesseract
from pathlib import Path
from .detector import ObjectDetector
from .ocr import TextRecognizer
from .utils import ImageUtils

class VisionSystemTester:
    def __init__(self):
        """Initialize test components."""
        self.detector = ObjectDetector()
        self.text_recognizer = TextRecognizer()
        self.image_utils = ImageUtils("test_outputs")
        
    def test_object_detection(self, image_path: str) -> bool:
        """Test object detection on a single image."""
        print(f"\nTesting object detection on: {image_path}")
        
        # Load test image
        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load test image")
            return False
            
        # Perform detection
        detections = self.detector.detect(image)
        
        # Draw results
        annotated = self.detector.draw_detections(image, detections)
        
        # Save result
        output_path = self.image_utils.save_image(annotated, "detection_test")
        if output_path:
            print(f"Detection results saved to: {output_path}")
            print(f"Objects detected: {len(detections.boxes)}")
            
            # Show confidence scores
            for box in detections.boxes.data:
                conf = box[4]
                cls = int(box[5])
                print(f"- {detections.names[cls]}: {conf:.2f}")
                
            return True
        return False
        
    def test_text_recognition(self, image_path: str) -> bool:
        """Test OCR on a single image."""
        print(f"\nTesting text recognition on: {image_path}")
        
        # Load test image
        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load test image")
            return False
        
        # Try multiple preprocessing methods for better results
        results = []
        
        # Method 1: Standard preprocessing
        preprocessed = self.text_recognizer.preprocess_image(image)
        text1 = self.text_recognizer.recognize_text(image)
        results.append(("Standard", text1))
        
        # Method 2: Enhanced preprocessing
        enhanced = self.image_utils.enhance_image_for_ocr(image)
        text2 = pytesseract.image_to_string(enhanced, config=r'--oem 3 --psm 6')
        results.append(("Enhanced", text2))
        
        # Method 3: Auto-rotated
        rotated, angle = self.image_utils.auto_rotate(image)
        if abs(angle) > 0.5:
            print(f"Auto-rotated image by {angle:.1f} degrees")
            text3 = self.text_recognizer.recognize_text(rotated)
            results.append(("Auto-rotated", text3))
        
        # Find best result (longest text)
        best_method, best_text = max(results, key=lambda x: len(x[1]))
        
        # Save preprocessed image
        output_path = self.image_utils.save_image(preprocessed, "ocr_test")
        
        if best_text:
            print(f"\nBest OCR Method: {best_method}")
            print("\nDetected Text:")
            print("-" * 50)
            print(best_text)
            print("-" * 50)
            print(f"Preprocessed image saved to: {output_path}")
            
            # Save all test images
            debug_dir = Path("ocr_debug")
            debug_dir.mkdir(exist_ok=True)
            
            cv2.imwrite(str(debug_dir / "original.jpg"), image)
            cv2.imwrite(str(debug_dir / "preprocessed.jpg"), preprocessed)
            cv2.imwrite(str(debug_dir / "enhanced.jpg"), enhanced)
            
            if abs(angle) > 0.5:
                cv2.imwrite(str(debug_dir / "rotated.jpg"), rotated)
            
            return True
        else:
            print("No text detected")
            return False