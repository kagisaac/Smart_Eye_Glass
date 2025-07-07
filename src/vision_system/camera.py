import cv2
import numpy as np
from typing import Tuple, Optional
import threading
import time

class Camera:
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (640, 480)):
        """Initialize camera capture optimized for Raspberry Pi."""
        self.camera_id = camera_id
        self.resolution = resolution
        self.camera = None
        self.is_opened = False
        
        # Performance optimization for Raspberry Pi
        self.buffer_size = 1
        self.fps_target = 15  # Lower FPS for better performance
        
        self._initialize_camera()

    def _initialize_camera(self):
        """Initialize camera with Raspberry Pi optimizations."""
        try:
            print(f"🔄 Initializing camera {self.camera_id}...")
            
            # Try different backends for Raspberry Pi compatibility
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    self.camera = cv2.VideoCapture(self.camera_id, backend)
                    if self.camera.isOpened():
                        print(f"✅ Camera opened with backend: {backend}")
                        break
                except Exception as e:
                    print(f"❌ Failed with backend {backend}: {str(e)}")
                    continue
            
            if not self.camera or not self.camera.isOpened():
                raise Exception("Could not open camera with any backend")
            
            # Set camera properties for Raspberry Pi optimization
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.fps_target)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # Additional optimizations for Raspberry Pi
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            print(f"📹 Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            self.is_opened = True
            
        except Exception as e:
            print(f"❌ Error initializing camera: {str(e)}")
            self.is_opened = False
            raise Exception(f"Failed to initialize camera: {str(e)}")

    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture a frame from the camera with error handling."""
        if not self.is_opened or not self.camera:
            return False, None
            
        try:
            # Clear buffer to get latest frame (important for Raspberry Pi)
            for _ in range(self.buffer_size):
                ret, frame = self.camera.read()
                if not ret:
                    break
            
            if ret and frame is not None:
                # Validate frame
                if frame.shape[0] > 0 and frame.shape[1] > 0:
                    return True, frame
                else:
                    print("❌ Invalid frame dimensions")
                    return False, None
            else:
                print("❌ Failed to capture frame")
                return False, None
                
        except Exception as e:
            print(f"❌ Error capturing frame: {str(e)}")
            return False, None

    def is_camera_opened(self) -> bool:
        """Check if camera is opened and working."""
        return self.is_opened and self.camera is not None and self.camera.isOpened()

    def get_camera_info(self) -> dict:
        """Get camera information."""
        if not self.is_camera_opened():
            return {}
            
        try:
            return {
                'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.camera.get(cv2.CAP_PROP_FPS),
                'backend': self.camera.getBackendName()
            }
        except Exception as e:
            print(f"❌ Error getting camera info: {str(e)}")
            return {}

    def release(self):
        """Release the camera resources."""
        try:
            if self.camera is not None:
                self.camera.release()
                print("📹 Camera released")
            self.is_opened = False
        except Exception as e:
            print(f"❌ Error releasing camera: {str(e)}")

