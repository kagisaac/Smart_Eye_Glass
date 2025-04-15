import pyttsx3
import threading
import queue
import time
from typing import Dict

class SpeechManager:
    def __init__(self):
        """Initialize text-to-speech engine with threading."""
        self.speech_queue = queue.Queue()
        self.is_running = True
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 175)  # Slightly faster rate
        self.engine.setProperty('volume', 1.0)  # Maximum volume
        self.last_speech_time = 0
        self.min_speech_interval = 0.5  # Reduced interval between speeches
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
        print("✅ Speech Manager initialized")

    def say(self, text: str, priority: bool = False):
        """Queue text for immediate speaking."""
        try:
            current_time = time.time()
            # Only queue new speech if enough time has passed or it's a priority message
            if priority or (current_time - self.last_speech_time >= self.min_speech_interval):
                self.speech_queue.put((text, priority))
                self.last_speech_time = current_time
        except Exception as e:
            print(f"❌ Error queueing speech: {str(e)}")

    def _speech_worker(self):
        """Background worker for speech processing."""
        while self.is_running:
            try:
                # Get text from queue with short timeout
                text, priority = self.speech_queue.get(timeout=0.1)
                
                # Clear queue if priority message
                if priority:
                    while not self.speech_queue.empty():
                        self.speech_queue.get_nowait()
                
                # Speak text
                print(f"🔊 Speaking: {text}")
                self.engine.say(text)
                self.engine.runAndWait()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in speech: {str(e)}")
                time.sleep(0.1)

    def queue_speech(self, text: str):
        """Alias for say method to maintain compatibility."""
        self.say(text, priority=False)

    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        if self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1.0)
        if hasattr(self, 'engine'):
            self.engine.stop()