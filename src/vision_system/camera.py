import cv2
from typing import Tuple, Optional

class Camera:
    def __init__(self, camera_id: int = 0):
        """Initialize camera capture."""
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise Exception("Could not open camera")

    def capture_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Capture a frame from the camera."""
        return self.camera.read()

    def release(self):
        """Release the camera resources."""
        if self.camera is not None:
            self.camera.release()