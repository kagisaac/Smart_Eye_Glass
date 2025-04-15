import cv2
import numpy as np
from .camera import Camera
from .detector import ObjectDetector
from .ocr import TextRecognizer
from .utils import ImageUtils

class VisionSystem:
    def __init__(self):
        """Initialize the vision system components."""
        self.camera = Camera()
        self.detector = ObjectDetector()
        self.text_recognizer = TextRecognizer()
        self.image_utils = ImageUtils()

    def run(self):
        """Run the main vision system loop."""
        try:
            print("\nVision System Started. Controls:")
            print("'q' - Quit")
            print("'s' - Save current frame")
            print("'t' - Perform OCR on current frame")
            print("'e' - Toggle enhanced mode (for better OCR)")
            print("-" * 50)
            
            enhanced_mode = False
            
            while True:
                # Capture frame
                ret, frame = self.camera.capture_frame()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Resize large images for better performance
                if frame.shape[1] > 1280:
                    frame = self.image_utils.resize_image(frame)
                
                # Perform object detection
                detections = self.detector.detect(frame)
                
                # Draw detections on frame
                annotated_frame = self.detector.draw_detections(frame, detections)
                
                # Apply enhancement in enhanced mode
                display_frame = annotated_frame
                if enhanced_mode:
                    # Create a split view showing original and enhanced
                    enhanced = self.image_utils.enhance_image_for_ocr(frame)
                    enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                    display_frame = np.hstack((annotated_frame, enhanced_color))
                    
                    # Resize if too large for display
                    if display_frame.shape[1] > 1280:
                        display_frame = self.image_utils.resize_image(display_frame)
                
                # Show the frame
                cv2.imshow('Vision System', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filepath = self.image_utils.save_image(frame)
                    if filepath:
                        print(f"\nFrame saved to: {filepath}")
                elif key == ord('t'):
                    # Try auto-rotation for better OCR results
                    rotated_frame, angle = self.image_utils.auto_rotate(frame)
                    if abs(angle) > 0.5:
                        print(f"\nAuto-rotated image by {angle:.1f} degrees for better OCR")
                        
                    # Perform OCR with enhanced preprocessing
                    text = self.text_recognizer.recognize_text(rotated_frame)
                    if text:
                        print("\nDetected Text:")
                        print("-" * 50)
                        print(text)
                        print("-" * 50)
                    else:
                        print("\nNo text detected")
                elif key == ord('e'):
                    enhanced_mode = not enhanced_mode
                    print(f"\nEnhanced mode: {'ON' if enhanced_mode else 'OFF'}")
                
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up system resources."""
        self.camera.release()
        cv2.destroyAllWindows()
        
    def process_image(self, image_path):
        """Process a single image file."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
            
        print(f"\nProcessing image: {image_path}")
        
        # Resize large images
        if image.shape[1] > 1280:
            image = self.image_utils.resize_image(image)
            
        # Detect objects
        detections = self.detector.detect(image)
        annotated = self.detector.draw_detections(image, detections)
        
        # Save detection results
        detection_path = self.image_utils.save_image(annotated, "detection")
        print(f"Detection results saved to: {detection_path}")
        
        # Print detection summary
        if hasattr(detections, 'boxes') and hasattr(detections.boxes, 'data'):
            boxes = detections.boxes.data
            print(f"Objects detected: {len(boxes)}")
            
            for box in boxes:
                conf = box[4]
                cls_id = int(box[5])
                cls_name = detections.names[cls_id] if hasattr(detections, 'names') else f"Class {cls_id}"
                print(f"- {cls_name}: {conf:.2f}")
        
        # Try auto-rotation for better OCR
        rotated, angle = self.image_utils.auto_rotate(image)
        if abs(angle) > 0.5:
            print(f"Auto-rotated image by {angle:.1f} degrees for better OCR")
            
        # Perform OCR with multiple preprocessing methods
        text = self.text_recognizer.recognize_text(rotated)
        
        if text:
            print("\nDetected Text:")
            print("-" * 50)
            print(text)
            print("-" * 50)
        else:
            print("\nNo text detected")
            
        # Show results
        cv2.imshow("Original", image)
        cv2.imshow("Detections", annotated)
        
        # Create enhanced version for OCR
        enhanced = self.image_utils.enhance_image_for_ocr(rotated)
        cv2.imshow("Enhanced for OCR", enhanced)
        
        print("\nPress any key to close the windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()