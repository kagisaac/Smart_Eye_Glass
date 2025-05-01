import cv2
import numpy as np
import time
import threading
import os
from paddleocr import PaddleOCR, draw_ocr
from langdetect import detect, LangDetectException
from typing import Optional, List, Dict, Tuple, Any
from .speech_manager import SpeechManager
import logging

class PaddleOCRRecognizer:
    """
    Text recognizer using PaddleOCR optimized for blind assistance with audio feedback.
    Specifically designed to handle Kinyarwanda text using English trained models.
    """
    def __init__(self, use_gpu: bool = False):
        """
        Initialize PaddleOCR with optimized settings for blind assistance.
        
        Args:
            use_gpu: Whether to use GPU acceleration (if available)
        """
        try:
            logging.info("🔄 Initializing PaddleOCR text recognition system...")
            
            # Initialize PaddleOCR with optimal settings for this use case
            # Using English model as fallback for Kinyarwanda
            self.ocr = PaddleOCR(
                use_angle_cls=True,  # Detect text orientation
                lang='en',           # English model (will adapt for Kinyarwanda)
                use_gpu=use_gpu,     # GPU usage based on availability
                show_log=False,      # Disable verbose logging
                # Optimization parameters for better detection
                rec_algorithm='CRNN',
                det_db_thresh=0.3,   # Lower threshold for better detection of low contrast text
                det_db_box_thresh=0.5,
                det_db_unclip_ratio=1.8,  # Higher value to better separate adjacent text
                rec_batch_num=6      # Batch processing for efficiency
            )
            
            # Initialize speech manager for audio feedback
            self.speech_manager = SpeechManager()
            
            # Text recognition state tracking
            self.is_processing = False
            self.should_stop = False
            self.processing_queue = []
            self.current_text = ""
            
            # Create a separate thread for processing
            self.processing_thread = None
            
            # Cache for recently recognized texts to avoid repetition
            self.text_cache = {}
            self.cache_expiry = 10  # seconds before cache entry expires
            
            logging.info("✅ PaddleOCR text recognition system initialized")
            
        except Exception as e:
            logging.error(f"❌ Error initializing PaddleOCR: {str(e)}")
            raise RuntimeError(f"Failed to initialize PaddleOCR: {str(e)}")

    def recognize_text(self, image: np.ndarray, is_capture: bool = False, 
                      priority: bool = False) -> Optional[str]:
        """
        Recognize text in image with enhanced accuracy using PaddleOCR.
        
        Args:
            image: Input image as numpy array
            is_capture: Whether this is from a manual capture (for feedback)
            priority: Whether to prioritize this recognition request
            
        Returns:
            Recognized text or None if no text found/error occurred
        """
        try:
            # Check if image is valid
            if image is None or image.size == 0:
                self.speech_manager.say("Invalid image for text recognition")
                return None
                
            if image.shape[0] < 20 or image.shape[1] < 20:
                self.speech_manager.say("Image is too small for text recognition")
                return None
            
            # Announce processing start for captured images
            if is_capture:
                self.speech_manager.say("Processing image for text", priority=True)
            
            # Apply image preprocessing to enhance text detection
            preprocessed_image = self._preprocess_image(image)
            
            # Set up processing
            self.should_stop = False
            self.is_processing = True
            
            # Run OCR with PaddleOCR
            start_time = time.time()
            results = self.ocr.ocr(preprocessed_image, cls=True)
            processing_time = time.time() - start_time
            
            if results is None or len(results) == 0 or not results[0]:
                if is_capture:
                    self.speech_manager.say("No text detected in image", priority=True)
                self.is_processing = False
                return None
            
            # Extract and process text results
            extracted_text = self._process_ocr_results(results[0])
            
            if not extracted_text:
                if is_capture:
                    self.speech_manager.say("No readable text found", priority=True)
                self.is_processing = False
                return None
            
            # Save the recognized text
            self.current_text = extracted_text
            
            # Add recognition timestamp to cache to avoid repetition
            cache_key = self._get_cache_key(extracted_text)
            self.text_cache[cache_key] = time.time()
            
            # Provide feedback on processing time for manual captures
            if is_capture:
                if processing_time > 2:
                    self.speech_manager.say(f"Text detected in {processing_time:.1f} seconds", priority=True)
                else:
                    self.speech_manager.say("Text detected", priority=True)
            
            # Speak the detected text with natural pauses between sentences
            self._speak_text_with_pauses(extracted_text, priority=priority)
            
            # Create annotated image for debugging/visualization if needed
            # self._save_annotated_image(image, results, extracted_text)
            
            return extracted_text
            
        except Exception as e:
            logging.error(f"❌ Error in text recognition: {str(e)}")
            if is_capture:
                self.speech_manager.say("Error processing text in image", priority=True)
            return None
        finally:
            self.is_processing = False
            self.should_stop = False

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve text recognition accuracy.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        try:
            # Create a copy to avoid modifying the original
            processed = image.copy()
            
            # Check if color image and convert to grayscale if needed
            if len(processed.shape) == 3:
                # Resize if image is too large (preserves aspect ratio)
                h, w = processed.shape[:2]
                max_dimension = 1280  # Maximum dimension for processing
                
                if max(h, w) > max_dimension:
                    scale = max_dimension / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    processed = cv2.resize(processed, (new_w, new_h), 
                                          interpolation=cv2.INTER_AREA)
                
                # Convert to LAB color space for better contrast enhancement
                lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_l = clahe.apply(l)
                
                # Merge back the channels
                enhanced_lab = cv2.merge([enhanced_l, a, b])
                processed = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                
                # Apply light denoising
                processed = cv2.fastNlMeansDenoisingColored(processed, None, 7, 7, 7, 21)
            
            return processed
            
        except Exception as e:
            logging.error(f"❌ Error in image preprocessing: {str(e)}")
            return image  # Return original if preprocessing fails

    def _process_ocr_results(self, results: List) -> str:
        """
        Process OCR results to extract and format text.
        
        Args:
            results: PaddleOCR detection results
            
        Returns:
            Processed text string
        """
        if not results:
            return ""
            
        text_blocks = []
        confidences = []
        
        # Extract text and confidence from results
        for line in results:
            if len(line) >= 2:  # Ensure result has both bbox and text/confidence
                text_info = line[1]
                if len(text_info) >= 2:  # Ensure text_info has both text and confidence
                    text = text_info[0]
                    confidence = text_info[1]
                    
                    # Filter out low confidence detections
                    if confidence > 0.6 and self._is_valid_text(text):
                        text_blocks.append(text)
                        confidences.append(confidence)
        
        # Join text blocks with appropriate spacing
        full_text = " ".join(text_blocks)
        
        # Post-process the text
        cleaned_text = self._post_process_text(full_text)
        
        return cleaned_text

    def _post_process_text(self, text: str) -> str:
        """
        Post-process recognized text to improve readability.
        
        Args:
            text: Raw recognized text
            
        Returns:
            Processed text
        """
        if not text:
            return ""
            
        # Replace multiple spaces with single space
        processed = " ".join(text.split())
        
        # Fix common OCR errors
        processed = processed.replace("l<", "k")
        processed = processed.replace("rn", "m")
        processed = processed.replace("vv", "w")
        
        # Fix sentence spacing
        for punct in ".!?":
            processed = processed.replace(f"{punct} ", f"{punct} ")
            processed = processed.replace(f"{punct}", f"{punct} ")
        
        # Remove excessive punctuation
        processed = processed.replace(",,", ",")
        processed = processed.replace("..", ".")
        
        return processed.strip()

    def _is_valid_text(self, text: str) -> bool:
        """
        Check if detected text is valid and meaningful.
        
        Args:
            text: Text to validate
            
        Returns:
            Whether text is valid
        """
        if not text or len(text.strip()) == 0:
            return False
            
        # Text should contain some letters (not just numbers or symbols)
        alpha_count = sum(c.isalpha() for c in text)
        if alpha_count < 1:
            return False
            
        # Check for minimum meaningful length
        if len(text.strip()) < 2:
            return False
            
        # Check that the text has a reasonable ratio of alphanumeric characters
        alphanum_ratio = sum(c.isalnum() for c in text) / len(text)
        if alphanum_ratio < 0.5:  # At least 50% should be alphanumeric
            return False
            
        return True

    def _speak_text_with_pauses(self, text: str, priority: bool = False):
        """
        Speak text with natural pauses between sentences.
        
        Args:
            text: Text to speak
            priority: Whether this speech should interrupt current speech
        """
        if not text:
            return
            
        # Split text into sentences for more natural reading
        sentences = self._split_into_sentences(text)
        
        # Check if this text has been recently read (avoid repetition)
        cache_key = self._get_cache_key(text)
        current_time = time.time()
        
        # Remove expired cache entries
        expired_keys = [k for k, v in self.text_cache.items() 
                       if current_time - v > self.cache_expiry]
        for k in expired_keys:
            del self.text_cache[k]
        
        # Skip if recently read (unless it's a priority request)
        if not priority and cache_key in self.text_cache and len(sentences) > 1:
            # Just read a brief summary for repeated text
            summary = f"Same text detected: {len(sentences)} sentences"
            self.speech_manager.say(summary)
            return
        
        # Begin reading text
        for i, sentence in enumerate(sentences):
            if self.should_stop:
                self.speech_manager.say("Reading interrupted", priority=True)
                break
                
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # Add longer pause after questions for more natural speech
            if sentence.strip().endswith("?"):
                pause_after = 0.8
            else:
                pause_after = 0.5
                
            # First sentence gets a "Reading text:" prefix if there are multiple sentences
            if i == 0 and len(sentences) > 1:
                self.speech_manager.say(f"Reading text: {sentence}", priority=priority)
            else:
                self.speech_manager.say(sentence, priority=priority)
                
            # Pause between sentences
            time.sleep(pause_after)

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for better reading pacing.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
            
        # Basic sentence splitting on punctuation
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in ".!?" and len(current.strip()) > 0:
                sentences.append(current.strip())
                current = ""
                
        # Add any remaining text
        if current.strip():
            sentences.append(current.strip())
            
        # Handle the case of very long sentences by splitting on commas too
        result = []
        for sentence in sentences:
            if len(sentence) > 100:  # Very long sentence
                parts = sentence.split(", ")
                if len(parts) > 1:
                    for i, part in enumerate(parts):
                        if i < len(parts) - 1:
                            result.append(part + ",")
                        else:
                            result.append(part)
                else:
                    result.append(sentence)
            else:
                result.append(sentence)
                
        return result

    def _get_cache_key(self, text: str) -> str:
        """
        Create a cache key for text to avoid repetition.
        
        Args:
            text: Input text
            
        Returns:
            Cache key
        """
        # Use first 50 chars as key, normalized to lowercase with spaces removed
        key = "".join(text.lower().split())[:50]
        return key

    def _save_annotated_image(self, image: np.ndarray, result, text: str):
        """
        Save annotated image with detected text regions for debugging.
        
        Args:
            image: Original image
            result: OCR result
            text: Recognized text
        """
        try:
            # Create debug directory if it doesn't exist
            debug_dir = "ocr_debug"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
                
            # Create annotated image
            boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
            
            # Draw annotations
            im_show = image.copy()
            im_show = draw_ocr(im_show, boxes, txts, scores)
            
            # Save the image
            timestamp = int(time.time())
            output_path = f"{debug_dir}/paddle_ocr_{timestamp}.jpg"
            cv2.imwrite(output_path, im_show)
            
            # Also save a text file with the recognized text
            with open(f"{debug_dir}/paddle_ocr_{timestamp}.txt", "w", encoding="utf-8") as f:
                f.write(text)
                
        except Exception as e:
            logging.error(f"❌ Error saving annotated image: {str(e)}")

    def detect_language(self, text: str) -> str:
        """
        Detect language of the text.
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        try:
            if not text or len(text) < 10:
                return "unknown"
                
            # Try to detect language
            lang = detect(text)
            return lang
        except LangDetectException:
            return "unknown"

    def interrupt(self):
        """Interrupt current text recognition/reading."""
        if self.is_processing:
            self.should_stop = True
            self.speech_manager.interrupt()
            logging.info("🛑 Interrupting text recognition")

    def cleanup(self):
        """Clean up resources."""
        self.should_stop = True
        if hasattr(self, 'speech_manager'):
            self.speech_manager.cleanup()