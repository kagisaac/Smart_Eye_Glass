import cv2
import numpy as np
import time
import os
import sys
import argparse
from typing import Any, List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Import detector and OCR modules
from vision_system.detector import ObjectDetector
from vision_system.paddle_ocr_text_recognizer import PaddleOCRRecognizer

class VisionAssistanceSystem:
    """Main vision assistance system for blind users with object detection and text recognition."""
    
    def __init__(self, camera_id: int = 0, model_path: str = 'yolov8s.pt', use_gpu: bool = False):
        """
        Initialize the vision assistance system.
        
        Args:
            camera_id: Camera device ID
            model_path: Path to YOLO model
            use_gpu: Whether to use GPU acceleration
        """
        try:
            print("\n========== VISION ASSISTANCE SYSTEM ==========")
            print("Initializing components...")
            
            # Initialize object detector
            self.detector = ObjectDetector(model_path)
            
            # Initialize text recognizer with PaddleOCR
            self.text_recognizer = PaddleOCRRecognizer(use_gpu=use_gpu)
            
            # Initialize camera
            self.camera = cv2.VideoCapture(camera_id)
            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open camera with ID {camera_id}")
                
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            # System state
            self.running = False
            self.paused = False
            self.text_mode = False
            self.last_key = None
            self.last_text_capture_time = 0
            self.text_capture_cooldown = 3.0  # seconds
            
            # Store the last recognized text for repeat functionality
            self.last_recognized_text = ""
            
            print("✅ All components initialized successfully")
            
        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            self.cleanup()
            raise

    def run(self):
        """Run the main processing loop."""
        try:
            self.running = True
            print("\n🚀 Starting vision assistance system")
            self.detector.speech_manager.say("Vision assistance system started. Press T for text mode, Spacebar to capture text, I to interrupt the Text, R to repeat text, and Q to quit.", priority=True, category="system")
            
            while self.running:
                # Process user input
                self._process_keyboard_input()
                
                # Skip frame processing if paused
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Read frame from camera
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    print("❌ Failed to capture frame, retrying...")
                    time.sleep(0.5)
                    continue
                
                # Process frame based on current mode
                if self.text_mode:
                    self._process_text_mode(frame)
                else:
                    self._process_object_detection_mode(frame)
                
                # Display status indicators
                self._display_status(frame)
                
                # Show frame
                cv2.imshow("Vision Assistance System", frame)
            
            print("\n👋 Vision assistance system stopped")
            
        except KeyboardInterrupt:
            print("\n👋 Vision assistance system interrupted")
        except Exception as e:
            print(f"\n❌ Error in main processing loop: {str(e)}")
        finally:
            self.cleanup()

    def _process_keyboard_input(self):
        """Process keyboard input."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Quit
            self.running = False
            self.detector.speech_manager.say("Shutting down", priority=True, category="system")
            
        elif key == ord('t'):  # Toggle text mode
            self.text_mode = not self.text_mode
            mode_name = "Text recognition mode" if self.text_mode else "Object detection mode"
            self.detector.speech_manager.say(f"Switched to {mode_name}", priority=True, category="system")
            
        elif key == ord(' ') and self.text_mode:  # Capture text in text mode
            current_time = time.time()
            if current_time - self.last_text_capture_time >= self.text_capture_cooldown:
                self.detector.speech_manager.say("Capturing image for text recognition", priority=True, category="system")
                self.last_text_capture_time = current_time
                self.last_key = key
                
        elif key == ord('p'):  # Pause/resume
            self.paused = not self.paused
            status = "paused" if self.paused else "resumed"
            self.detector.speech_manager.say(f"System {status}", priority=True, category="system")
            
        elif key == ord('i'):  # Interrupt text reading
            self.text_recognizer.interrupt()
            
        elif key == ord('r'):  # Repeat last text
            self._repeat_last_text()
            
        self.last_key = key if key != 255 else self.last_key

    def _repeat_last_text(self):
        """Repeat the last recognized text."""
        if self.last_recognized_text:
            print(f"🔄 Repeating text: {self.last_recognized_text}")
            self.detector.speech_manager.say("Repeating last text", priority=True, category="system")
            self.detector.speech_manager.announce_text(self.last_recognized_text)
        else:
            self.detector.speech_manager.say("No text to repeat", priority=True, category="system")

    def _process_object_detection_mode(self, frame: np.ndarray):
        """
        Process frame in object detection mode.
        
        Args:
            frame: Current camera frame
        """
        # Detect objects
        detections = self.detector.detect(frame)
        
        # Draw detections and get detection info
        if detections is not None:
            frame, significant_detections = self.detector.draw_detections(frame, detections)
            
            # Draw enhanced annotated bounding boxes
            self._draw_annotated_bounding_boxes(frame, detections)
            
            # Announce detections through the detector
            self.detector.announce_detections(significant_detections)

    def _draw_annotated_bounding_boxes(self, frame: np.ndarray, detections):
        """
        Draw enhanced annotated bounding boxes with labels and confidence scores.
        
        Args:
            frame: Current camera frame
            detections: Detection results from the object detector
        """
        # Process each detection based on the format of your detection model
        # The implementation will depend on the exact structure of your detection results
        
        # For YOLOv8 format
        if hasattr(detections, 'boxes'):
            boxes = detections.boxes.cpu().numpy()
            
            for i, box in enumerate(boxes):
                # Extract box information
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = detections.names[cls]
                
                # Skip low confidence detections
                if conf < 0.5:
                    continue
                
                # Calculate colors based on confidence
                color_value = min(255, int(conf * 255))
                box_color = (0, color_value, 255 - color_value)  # RGB: red to green based on confidence
                
                # Draw filled rectangle with transparency for better visibility
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
                frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
                
                # Draw bounding box border
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Create label with class name and confidence
                label = f"{cls_name}: {conf:.2f}"
                
                # Determine best position for text (top of box)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_bg_y1 = max(0, y1 - text_size[1] - 10)
                text_bg_y2 = y1
                
                # Draw background for text
                cv2.rectangle(frame, (x1, text_bg_y1), (x1 + text_size[0] + 10, text_bg_y2), box_color, -1)
                
                # Draw text
                cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def _process_text_mode(self, frame: np.ndarray):
        """
        Process frame in text recognition mode.
        
        Args:
            frame: Current camera frame
        """
        # Draw text mode indicator
        cv2.putText(frame, "TEXT MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Process capture request
        if self.last_key == ord(' '):
            self.last_key = None  # Reset key state
            
            # Capture frame for text recognition
            # Add visual indicator that capture is happening
            flash_frame = frame.copy()
            cv2.rectangle(flash_frame, (0, 0), (flash_frame.shape[1], flash_frame.shape[0]), 
                         (255, 255, 255), 15)
            cv2.imshow("Vision Assistance System", flash_frame)
            cv2.waitKey(100)
            
            # Process text in a non-blocking way
            self._process_text_recognition(frame.copy())

    def _process_text_recognition(self, frame: np.ndarray):
        """
        Process text recognition on a captured frame.
        
        Args:
            frame: Frame to process
        """
        try:
            # Save captured image for debugging if needed
            self._save_captured_image(frame)
            
            # Recognize text with priority
            text = self.text_recognizer.recognize_text(frame, is_capture=True, priority=True)
            
            if text:
                print(f"📝 Recognized text: {text}")
                # Store the text for repeat functionality
                self.last_recognized_text = text
                # Use the speech manager to announce the text
                self.detector.speech_manager.announce_text(text)
            else:
                self.detector.speech_manager.say("No text detected", priority=True, category="text")
        except Exception as e:
            print(f"❌ Error in text recognition: {str(e)}")
            self.detector.speech_manager.say("Error processing text", priority=True, category="system")

    def _save_captured_image(self, frame: np.ndarray):
        """
        Save captured image for debugging.
        
        Args:
            frame: Frame to save
        """
        try:
            # Create directory if it doesn't exist
            save_dir = "captured_images"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # Save image with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{save_dir}/capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
        except Exception as e:
            print(f"❌ Error saving captured image: {str(e)}")

    def _display_status(self, frame: np.ndarray):
        """
        Display status indicators on frame.
        
        Args:
            frame: Frame to annotate
        """
        # Add mode indicator
        mode_text = "TEXT MODE" if self.text_mode else "OBJECT DETECTION MODE"
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 0, 255) if self.text_mode else (0, 255, 0), 2, cv2.LINE_AA)
        
        # Add controls reminder
        controls = "T: Toggle Mode | SPACE: Capture Text | R: Repeat Text | Q: Quit"
        cv2.putText(frame, controls, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add paused indicator if needed
        if self.paused:
            cv2.putText(frame, "PAUSED", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 0, 255), 2, cv2.LINE_AA)

    def cleanup(self):
        """Clean up resources."""
        # Release camera
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.release()
            
        # Clean up detector
        if hasattr(self, 'detector'):
            self.detector.cleanup()
            
        # Clean up text recognizer
        if hasattr(self, 'text_recognizer'):
            self.text_recognizer.cleanup()
            
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        print("✅ Cleanup completed")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Vision Assistance System for Blind Users")
    
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')
    
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='Path to YOLO model (default: yolov8s.pt)')
    
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration for OCR (default: False)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Create and run vision assistance system
        system = VisionAssistanceSystem(
            camera_id=args.camera,
            model_path=args.model,
            use_gpu=args.gpu
        )
        
        # Run the system
        system.run()
        
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())