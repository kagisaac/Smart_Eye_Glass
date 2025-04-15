from vision_system.tests import VisionSystemTester
from pathlib import Path

def main():
    # Create test images directory if it doesn't exist
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    print("\nVision System Test Suite")
    print("=" * 50)
    print("\nBefore running tests, please ensure you have:")
    print("1. A test image with objects (e.g., people, cars, etc.) in 'test_images/objects.jpg'")
    print("2. A test image with clear text in 'test_images/text.jpg'")
    print("\nTest images should be placed in the 'test_images' directory.")
    
    input("\nPress Enter to start testing...")
    
    tester = VisionSystemTester()
    
    # Test object detection
    object_image = test_dir / "objects.jpg"
    if object_image.exists():
        tester.test_object_detection(str(object_image))
    else:
        print(f"\nWarning: Object test image not found at {object_image}")
        print("Please add an image with objects to test detection.")
    
    # Test text recognition
    text_image = test_dir / "text.jpg"
    if text_image.exists():
        tester.test_text_recognition(str(text_image))
    else:
        print(f"\nWarning: Text test image not found at {text_image}")
        print("Please add an image with text to test OCR.")

if __name__ == "__main__":
    main()