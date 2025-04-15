import os
from datetime import datetime
import cv2
import numpy as np
from typing import Optional, Tuple

class ImageUtils:
    def __init__(self, output_dir: str = "captured_images"):
        """Initialize utilities with output directory."""
        self.output_dir = output_dir
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_image(self, image: np.ndarray, prefix: str = "capture") -> Optional[str]:
        """Save image with timestamp."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, image)
            return filepath
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return None
            
    def enhance_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced image enhancement techniques for OCR."""
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
        
    def resize_image(self, image: np.ndarray, target_width: int = 1280) -> np.ndarray:
        """Resize image while maintaining aspect ratio."""
        h, w = image.shape[:2]
        ratio = target_width / w
        new_h = int(h * ratio)
        return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)
        
    def auto_rotate(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Automatically rotate image to correct orientation for OCR."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return image, 0.0
            
        # Calculate angles of lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:  # Avoid division by zero
                continue
            angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
            angles.append(angle)
            
        if not angles:
            return image, 0.0
            
        # Get median angle
        median_angle = np.median(angles)
        
        # If angle is close to horizontal, adjust it
        if abs(median_angle) < 0.5:
            return image, 0.0
            
        # Rotate image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated, median_angle