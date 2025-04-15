import cv2
import numpy as np
from vision_system.detector import ObjectDetector
from vision_system.text_recognizer import TextRecognizer
import time

def main():
    """Main function for real-time object detection and text recognition."""
    try:
        # Initialize components
        detector = ObjectDetector(model_path='yolov8s.pt')
        recognizer = TextRecognizer()
        
        # Initialize video capture with HD resolution
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam.")
            return
            
        # Announce system start
        detector.speech_manager.say("Vision system started. Press T for text recognition mode, O for object detection mode.")
        
        print("\n🎥 Vision System Started. Controls:")
        print("'q' - Quit")
        print("'s' - Save current frame")
        print("'t' - Toggle text recognition mode")
        print("'o' - Toggle object detection mode")
        print("-" * 50)
        
        # Mode flags
        ocr_mode = False
        object_mode = True
        
        while True:
            # Read frame in HD
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            display_frame = frame.copy()
            
            # Handle modes
            if object_mode and not ocr_mode:
                detections = detector.detect(frame)
                display_frame, _ = detector.draw_detections(display_frame, detections)
            elif ocr_mode and not object_mode:
                text = recognizer.recognize_text(frame)
                if text:
                    cv2.putText(display_frame, "Text Recognition Active", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show mode status
            status = []
            if object_mode:
                status.append("Object Detection Mode")
            if ocr_mode:
                status.append("Text Recognition Mode")
            
            # Display status
            y_pos = 30
            for stat in status:
                cv2.putText(display_frame, stat, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_pos += 25
            
            # Show frame
            cv2.imshow("📷 Vision System", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                detector.speech_manager.say("Shutting down vision system")
                print("\n👋 Quitting...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                detector.speech_manager.say("Frame saved")
                print(f"💾 Frame saved as '{filename}'")
            elif key == ord('t'):
                ocr_mode = not ocr_mode
                object_mode = not ocr_mode  # Ensure only one mode is active
                mode_msg = "Switching to text recognition mode" if ocr_mode else "Switching to object detection mode"
                detector.speech_manager.say(mode_msg)
                print(f"🔍 OCR Mode: {'ON' if ocr_mode else 'OFF'}")
            elif key == ord('o'):
                object_mode = not object_mode
                ocr_mode = not object_mode  # Ensure only one mode is active
                mode_msg = "Switching to object detection mode" if object_mode else "Switching to text recognition mode"
                detector.speech_manager.say(mode_msg)
                print(f"👁️ Object Detection: {'ON' if object_mode else 'OFF'}")
            
    except Exception as e:
        print(f"❌ Error in main loop: {str(e)}")
        detector.speech_manager.say("An error occurred. System shutting down.")
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()
        recognizer.cleanup()
        print("✅ Vision system ended.")

if __name__ == "__main__":
    main()