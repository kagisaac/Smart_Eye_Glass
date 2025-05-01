import cv2
import numpy as np
from typing import Any, List, Dict, Tuple, Optional
import logging
from ultralytics import YOLO
import time

# Import SpeechManager
from vision_system.speech_manager import SpeechManager

class ObjectDetector:
    """Object detector using YOLO with speech feedback for blind users."""
    
    def __init__(self, model_path: str = 'yolov8s.pt'):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to YOLO model
        """
        try:
            print("🔄 Loading YOLO model...")
            # Initialize YOLO model
            self.model = YOLO(model_path)
            
            # Initialize speech manager for announcements
            self.speech_manager = SpeechManager()
            
            # Detection settings
            self.confidence_threshold = 0.45
            self.min_announcement_interval = 2.0  # seconds
            self.last_announcement_time = 0
            self.previous_detections = []
            self.detection_history = {}
            
            print("✅ YOLO model loaded.")
            
        except Exception as e:
            print(f"❌ Failed to initialize object detector: {str(e)}")
            raise
    
    def detect(self, frame: np.ndarray) -> Optional[List[Dict]]:
        """
        Detect objects in frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detection dictionaries or None on error
        """
        try:
            # Run detection
            results = self.model(frame, conf=self.confidence_threshold)[0]
            
            # Format results
            detections = []
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = result
                
                # Get class name
                class_name = results.names[int(class_id)]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class': class_name
                })
                
            return detections
            
        except Exception as e:
            logging.error(f"Detection error: {str(e)}")
            return None
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Draw enhanced bounding boxes and labels for detections.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            
        Returns:
            Tuple of (annotated frame, significant detections)
        """
        significant_detections = []
        
        # Create a copy of the frame
        annotated_frame = frame.copy()
        
        # Draw each detection
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Skip detections with low confidence
            if confidence < self.confidence_threshold:
                continue
                
            # Add to significant detections
            significant_detections.append(detection)
            
            # Extract box coordinates
            x1, y1, x2, y2 = bbox
            
            # Calculate color based on confidence
            # Higher confidence: more green, less red
            green = int(min(confidence * 2, 1.0) * 255)
            red = int((1.0 - min(confidence, 0.8)) * 255)
            color = (0, green, red)  # BGR format
            
            # Draw semi-transparent filled rectangle for better visibility
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # Filled rectangle
            annotated_frame = cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0)  # Blend with 0.2 opacity
            
            # Draw solid bounding box border
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label with class and confidence
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Calculate label position (above the box if possible)
            text_y = max(0, y1 - 5)
            
            # Draw background for label text (fully filled rectangle)
            cv2.rectangle(annotated_frame, 
                         (x1, text_y - text_size[1] - baseline), 
                         (x1 + text_size[0] + 10, text_y + baseline), 
                         color, cv2.FILLED)
            
            # Draw label text (in white for better contrast)
            cv2.putText(annotated_frame, label, (x1 + 5, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Return annotated frame and significant detections
        return annotated_frame, significant_detections
    
    def announce_detections(self, detections: List[Dict]):
        """
        Announce significant changes in detections.
        
        Args:
            detections: List of detection dictionaries
        """
        current_time = time.time()
        
        # Skip if not enough time has passed since last announcement
        if current_time - self.last_announcement_time < self.min_announcement_interval:
            return
            
        # Skip if no detections
        if not detections:
            return
            
        # Update detection history
        current_classes = {}
        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Count only if confidence is high enough
            if confidence >= self.confidence_threshold:
                if class_name in current_classes:
                    current_classes[class_name] += 1
                else:
                    current_classes[class_name] = 1
        
        # Check what's new compared to previous announcement
        new_classes = {}
        for class_name, count in current_classes.items():
            # Class is new or count increased significantly
            if (class_name not in self.detection_history or 
                count > self.detection_history.get(class_name, 0) + 1):
                new_classes[class_name] = count
        
        # Prepare announcement
        if new_classes:
            announcement_parts = []
            
            for class_name, count in new_classes.items():
                if count == 1:
                    announcement_parts.append(f"1 {class_name}")
                else:
                    announcement_parts.append(f"{count} {class_name}s")
            
            # Format natural language announcement
            if len(announcement_parts) == 1:
                announcement = announcement_parts[0]
            elif len(announcement_parts) == 2:
                announcement = f"{announcement_parts[0]} and {announcement_parts[1]}"
            else:
                announcement = ", ".join(announcement_parts[:-1]) + f", and {announcement_parts[-1]}"
            
            # Announce new detections
            self.speech_manager.say(f"Detected {announcement}", priority=True, category="object")
            
            # Update state
            self.last_announcement_time = current_time
            self.detection_history = current_classes
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Clean up speech manager
            if hasattr(self, 'speech_manager'):
                self.speech_manager.cleanup()
        except Exception as e:
            logging.error(f"Error during detector cleanup: {str(e)}")