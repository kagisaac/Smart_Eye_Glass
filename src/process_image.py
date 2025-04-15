import cv2
import sys
import os
from vision_system import VisionSystem

def main():
    """Process a single image with the vision system."""
    if len(sys.argv) < 2:
        print("Usage: python process_image.py <image_path>")
        return
        
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
        
    try:
        # Initialize vision system
        vision_system = VisionSystem()
        
        # Process the image
        vision_system.process_image(image_path)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()