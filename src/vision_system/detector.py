import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import Any, List, Dict, Tuple
from .speech_manager import SpeechManager

class ObjectDetector:
    def __init__(self, model_path: str = 'yolov8s.pt'):
        """Initialize YOLO model and speech manager."""
        try:
            print("\n🔄 Loading YOLO model...")
            self.model = YOLO(model_path)
            # Optimized detection settings
            self.model.conf = 0.45
            self.model.iou = 0.45
            self.model.max_det = 300
            
            # Initialize speech manager
            self.speech_manager = SpeechManager()
            
            # Detection state tracking
            self.last_announcement = {}
            self.announcement_cooldown = 1.0  # seconds
            self.last_announcement_time = 0
            
            print("✅ YOLO model loaded.")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {str(e)}")
            raise Exception(f"Failed to initialize YOLO model: {str(e)}")

    def detect(self, frame: np.ndarray) -> Any:
        """Detect objects in frame."""
        try:
            # Enhance image for better detection
            enhanced_frame = self._enhance_image(frame)
            
            # Run detection
            results = self.model(enhanced_frame, verbose=False)
            return results[0]
        except Exception as e:
            print(f"❌ Error in detection: {str(e)}")
            return None

    def draw_detections(self, frame: np.ndarray, detections: Any) -> Tuple[np.ndarray, List[Dict]]:
        """Draw detections on frame and handle speech output."""
        if not hasattr(detections, 'boxes') or not detections.boxes.data.any():
            return frame, []

        detection_info = []
        current_detections = {}
        current_time = time.time()

        # Process all detections first
        for box in detections.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_id = int(cls)
            
            # Get class name
            cls_name = detections.names[cls_id] if hasattr(detections, 'names') else f"Class {cls_id}"
            
            # Update current detections count
            current_detections[cls_name] = current_detections.get(cls_name, 0) + 1
            
            # Draw detection
            color = self._get_color(cls_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence
            label = f"{cls_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add to detection info
            detection_info.append({
                'class': cls_name,
                'confidence': float(conf),
                'bbox': (x1, y1, x2, y2)
            })

        # Check if we should make a new announcement
        if current_time - self.last_announcement_time >= self.announcement_cooldown:
            # Compare with last announcement to only announce changes
            announcement_parts = []
            for cls_name, count in current_detections.items():
                if cls_name not in self.last_announcement or self.last_announcement[cls_name] != count:
                    if count == 1:
                        announcement_parts.append(f"one {cls_name}")
                    else:
                        announcement_parts.append(f"{count} {cls_name}s")
            
            if announcement_parts:
                announcement = "Detected " + ", and ".join(announcement_parts)
                self.speech_manager.say(announcement, priority=True)
                self.last_announcement = current_detections.copy()
                self.last_announcement_time = current_time

        return frame, detection_info

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better detection."""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except Exception as e:
            print(f"❌ Error enhancing image: {str(e)}")
            return image

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for visualization."""
        colors = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
        ]
        return colors[class_id % len(colors)]

    def cleanup(self):
        """Clean up resources."""
        self.speech_manager.cleanup()
        