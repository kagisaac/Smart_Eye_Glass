import pyttsx3
import threading
import queue
import time
from typing import Dict, Optional, List
import logging

class SpeechManager:
    """
    Speech manager for vision assistance system with priority, interruption,
    and voice customization features.
    """
    
    def __init__(self, rate: int = 155, volume: float = 1.0, voice: Optional[str] = 1):
        """
        Initialize text-to-speech engine with threading.
        
        Args:
            rate: Speech rate (words per minute)
            volume: Speech volume (0.0 to 1.0)
            voice: Voice ID to use (None for default)
        """
        try:
            # Initialize queue and state
            self.speech_queue = queue.Queue()
            self.is_running = True
            self.is_speaking = False
            self.current_speech_id = 0
            self.interrupt_requested = False
            
            # Initialize TTS engine
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            
            # Set voice if specified
            if voice:
                self.set_voice(voice)
            else:
                # Try to find a high-quality voice
                self._set_best_available_voice()
            
            # Speech timing control
            self.last_speech_time = 0
            self.min_speech_interval = 0.3  # Reduced interval for more responsive feedback
            
            # Start speech worker thread
            self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.speech_thread.start()
            
            # Store available voices for later use
            self.available_voices = self.engine.getProperty('voices')
            
            logging.info("Speech Manager initialized successfully")
            print("✅ Speech Manager initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize Speech Manager: {str(e)}")
            print(f"❌ Failed to initialize Speech Manager: {str(e)}")
            raise
    
    def say(self, text: str, priority: bool = False, interrupt: bool = False, category: str = "general"):
        """
        Queue text for speaking with optional priority and interruption.
        
        Args:
            text: Text to speak
            priority: If True, this speech jumps to front of queue
            interrupt: If True, interrupts current speech
            category: Speech category for filtering (e.g., "object", "text", "system")
        """
        try:
            if not text or not isinstance(text, str):
                return
                
            # Clean up text for better speech synthesis
            text = self._preprocess_text(text)
            
            current_time = time.time()
            should_queue = (
                priority or 
                interrupt or 
                (current_time - self.last_speech_time >= self.min_speech_interval)
            )
            
            if should_queue:
                speech_id = self.current_speech_id + 1
                self.current_speech_id = speech_id
                
                if interrupt and self.is_speaking:
                    # Signal interruption and add new speech
                    self.interrupt_requested = True
                    self.speech_queue.put((text, priority, speech_id, category))
                    self.last_speech_time = current_time
                elif priority:
                    # For priority without interruption, use a temporary queue
                    temp_queue = queue.Queue()
                    temp_queue.put((text, priority, speech_id, category))
                    
                    # Then add all existing items back
                    while not self.speech_queue.empty():
                        try:
                            item = self.speech_queue.get_nowait()
                            temp_queue.put(item)
                        except queue.Empty:
                            break
                    
                    # Swap queues
                    self.speech_queue = temp_queue
                    self.last_speech_time = current_time
                else:
                    # Normal queueing
                    self.speech_queue.put((text, priority, speech_id, category))
                    self.last_speech_time = current_time
                    
        except Exception as e:
            logging.error(f"Error queueing speech: {str(e)}")
            print(f"❌ Error queueing speech: {str(e)}")
    
    def announce_objects(self, objects: List[Dict]):
        """
        Create a summary announcement for detected objects.
        
        Args:
            objects: List of detected objects with class and confidence
        """
        if not objects:
            return
            
        # Filter out low confidence detections
        significant_objects = [obj for obj in objects if obj.get('confidence', 0) > 0.4]
        
        if not significant_objects:
            return
            
        # Count objects by class
        object_counts = {}
        for obj in significant_objects:
            obj_class = obj.get('class', '').lower()
            if obj_class:
                object_counts[obj_class] = object_counts.get(obj_class, 0) + 1
        
        # Create announcement text
        announcement_parts = []
        for obj_class, count in object_counts.items():
            if count == 1:
                announcement_parts.append(f"1 {obj_class}")
            else:
                announcement_parts.append(f"{count} {obj_class}s")
        
        if announcement_parts:
            if len(announcement_parts) == 1:
                announcement = announcement_parts[0]
            elif len(announcement_parts) == 2:
                announcement = f"{announcement_parts[0]} and {announcement_parts[1]}"
            else:
                announcement = ", ".join(announcement_parts[:-1]) + f", and {announcement_parts[-1]}"
                
            self.say(f"Detected {announcement}", category="object")
    
    def announce_text(self, text: str):
        """
        Announce recognized text with appropriate formatting.
        
        Args:
            text: Recognized text to announce
        """
        if not text:
            self.say("No text detected", category="text")
            return
            
        # Clean up text for better reading
        clean_text = self._preprocess_text(text)
        
        if len(clean_text) > 500:
            # For long text, provide a summary first
            words = len(clean_text.split())
            self.say(f"Found text with approximately {words} words. Reading...", 
                     priority=True, category="text")
            time.sleep(0.5)  # Brief pause
        
        self.say(clean_text, priority=True, category="text")
    
    def interrupt(self):
        """Interrupt current speech."""
        self.interrupt_requested = True
        
        # Clear the queue
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
    
    def _speech_worker(self):
        """Background worker for speech processing."""
        while self.is_running:
            try:
                # Check if we need to break from current speech
                if self.interrupt_requested:
                    if self.is_speaking:
                        try:
                            self.engine.stop()
                        except:
                            pass
                    self.is_speaking = False
                    self.interrupt_requested = False
                    time.sleep(0.1)
                    continue
                
                # Get text from queue with short timeout
                text, priority, speech_id, category = self.speech_queue.get(timeout=0.1)
                
                # Speak text
                logging.info(f"Speaking ({category}): {text}")
                print(f"🔊 Speaking ({category}): {text}")
                
                self.is_speaking = True
                self.engine.say(text)
                self.engine.runAndWait()
                self.is_speaking = False
                
            except queue.Empty:
                time.sleep(0.01)  # Short sleep to prevent CPU hogging
                continue
            except Exception as e:
                logging.error(f"Error in speech worker: {str(e)}")
                print(f"❌ Error in speech: {str(e)}")
                self.is_speaking = False
                time.sleep(0.1)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better speech synthesis.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
            
        # Replace common symbols for better reading
        replacements = {
            '%': ' percent',
            '&': ' and ',
            '+': ' plus ',
            '=': ' equals ',
            '@': ' at ',
            '#': ' number ',
            '>': ' greater than ',
            '<': ' less than ',
            '...': ', etc',
            '…': ', etc',
        }
        
        result = text
        for symbol, replacement in replacements.items():
            result = result.replace(symbol, replacement)
            
        # Remove excessive whitespace
        result = ' '.join(result.split())
        
        return result
    
    def set_voice(self, voice_id: str) -> bool:
        """
        Set voice by ID.
        
        Args:
            voice_id: Voice ID to use
            
        Returns:
            Success status
        """
        try:
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if voice.id == voice_id:
                    self.engine.setProperty('voice', voice.id)
                    return True
            return False
        except Exception as e:
            logging.error(f"Error setting voice: {str(e)}")
            return False
    
    def set_rate(self, rate: int) -> bool:
        """
        Set speech rate.
        
        Args:
            rate: Speech rate (words per minute)
            
        Returns:
            Success status
        """
        try:
            self.engine.setProperty('rate', rate)
            return True
        except Exception as e:
            logging.error(f"Error setting speech rate: {str(e)}")
            return False
    
    def set_volume(self, volume: float) -> bool:
        """
        Set speech volume.
        
        Args:
            volume: Speech volume (0.0 to 1.0)
            
        Returns:
            Success status
        """
        try:
            volume = max(0.0, min(1.0, volume))  # Clamp to valid range
            self.engine.setProperty('volume', volume)
            return True
        except Exception as e:
            logging.error(f"Error setting speech volume: {str(e)}")
            return False
    
    def _set_best_available_voice(self):
        """Set the best available voice for screen reading."""
        try:
            voices = self.engine.getProperty('voices')
            
            # Preferred voice keywords (in order of preference)
            preferred_keywords = [ 'david', 'hazel', 'english', 'us', 'uk']
            
            for keyword in preferred_keywords:
                for voice in voices:
                    if keyword.lower() in voice.name.lower() or keyword.lower() in voice.id.lower():
                        self.engine.setProperty('voice', voice.id)
                        logging.info(f"Set voice to {voice.name}")
                        return
            
            # If no preferred voice found, use the first available
            if voices:
                self.engine.setProperty('voice', voices[0].id)
                logging.info(f"Defaulted to voice {voices[0].name}")
                
        except Exception as e:
            logging.error(f"Error setting best voice: {str(e)}")
    
    def get_available_voices(self) -> List[Dict]:
        """
        Get list of available voices.
        
        Returns:
            List of voice dictionaries with 'id' and 'name'
        """
        try:
            voices = self.engine.getProperty('voices')
            return [{'id': voice.id, 'name': voice.name} for voice in voices]
        except Exception as e:
            logging.error(f"Error getting voices: {str(e)}")
            return []
    
    def queue_speech(self, text: str):
        """
        Alias for say method to maintain compatibility with older code.
        
        Args:
            text: Text to speak
        """
        self.say(text, priority=False)
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.is_running = False
            self.interrupt_requested = True
            
            if hasattr(self, 'engine'):
                try:
                    self.engine.stop()
                except:
                    pass
            
            if hasattr(self, 'speech_thread') and self.speech_thread.is_alive():
                self.speech_thread.join(timeout=1.0)
                
            logging.info("Speech Manager cleaned up")
        except Exception as e:
            logging.error(f"Error during Speech Manager cleanup: {str(e)}")