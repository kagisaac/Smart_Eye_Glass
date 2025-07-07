import cv2
import numpy as np
import time
import threading
import os
from paddleocr import PaddleOCR
from langdetect import detect, LangDetectException
from typing import Optional, List, Dict, Tuple, Any
from .speech_manager import SpeechManager
import logging
from scipy import ndimage
import skimage.filters as filters
from skimage import morphology, exposure, restoration
from skimage.util import img_as_ubyte

class PaddleOCRRecognizer:
    """
    Enhanced text recognizer using PaddleOCR with advanced image preprocessing.
    Specifically designed to handle various image conditions and improve recognition accuracy.
    """
    def __init__(self, use_gpu: bool = False):
        """
        Initialize PaddleOCR with optimized settings for blind assistance.
        
        Args:
            use_gpu: Whether to use GPU acceleration (if available)
        """
        try:
            logging.info("🔥 Initializing Enhanced PaddleOCR text recognition system...")
            
            # Initialize PaddleOCR with basic parameters first
            # Remove unsupported parameters that are causing issues
            ocr_params = {
                'use_angle_cls': True,
                'lang': 'en',
                #'show_log': False  # Reduce verbose logging
            }
            
            # Add GPU parameter if requested
            if use_gpu:
                ocr_params['use_gpu'] = True
            
            # Initialize OCR with basic parameters only
            self.ocr = PaddleOCR(**ocr_params)
            logging.info(f"✅ PaddleOCR initialized successfully")
            
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
            
            logging.info("✅ Enhanced PaddleOCR text recognition system initialized")
            
        except Exception as e:
            logging.error(f"❌ Error initializing PaddleOCR: {str(e)}")
            raise RuntimeError(f"Failed to initialize PaddleOCR: {str(e)}")

    def recognize_text(self, image: np.ndarray, is_capture: bool = False, 
                      priority: bool = False) -> Optional[str]:
        """
        Recognize text in image with enhanced preprocessing and multiple attempts.
        
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
            
            # Set up processing
            self.should_stop = False
            self.is_processing = True
            
            # Try multiple preprocessing approaches
            best_result = None
            best_confidence = 0
            
            start_time = time.time()
            
            # List of preprocessing strategies to try
            preprocessing_strategies = [
                'enhanced_default',
                'high_contrast',
                'adaptive_threshold',
                'edge_enhanced',
                'denoised',
                'gamma_corrected',
                'morphological_enhanced'
            ]
            
            for strategy in preprocessing_strategies:
                if self.should_stop:
                    break
                    
                try:
                    # Apply preprocessing strategy
                    preprocessed_image = self._preprocess_image_advanced(image, strategy)
                    
                    # Run OCR without the problematic 'cls' parameter
                    # The 'cls' parameter is already set during initialization
                    results = self.ocr.ocr(preprocessed_image)
                    
                    if results and len(results) > 0 and results[0]:
                        # Extract and process text results
                        extracted_text, avg_confidence = self._process_ocr_results_with_confidence(results[0])
                        
                        # Keep the best result
                        if extracted_text and avg_confidence > best_confidence:
                            best_result = extracted_text
                            best_confidence = avg_confidence
                            
                        # If we got a very good result, stop trying other methods
                        if avg_confidence > 0.85:
                            break
                            
                except Exception as e:
                    logging.warning(f"⚠️ Error with preprocessing strategy {strategy}: {str(e)}")
                    continue
            
            processing_time = time.time() - start_time
            
            if not best_result:
                # Try one more time with original image and minimal preprocessing
                try:
                    logging.info("🔄 Trying with minimal preprocessing...")
                    simple_processed = self._simple_preprocessing(image)
                    results = self.ocr.ocr(simple_processed)
                    
                    if results and len(results) > 0 and results[0]:
                        extracted_text, avg_confidence = self._process_ocr_results_with_confidence(results[0])
                        if extracted_text:
                            best_result = extracted_text
                            best_confidence = avg_confidence
                            logging.info(f"✅ Success with minimal preprocessing: {best_confidence:.2f}")
                            
                except Exception as e:
                    logging.warning(f"⚠️ Error with minimal preprocessing: {str(e)}")
            
            if not best_result:
                if is_capture:
                    self.speech_manager.say("No readable text found despite advanced processing", priority=True)
                self.is_processing = False
                return None
            
            # Save the recognized text
            self.current_text = best_result
            
            # Add recognition timestamp to cache to avoid repetition
            cache_key = self._get_cache_key(best_result)
            self.text_cache[cache_key] = time.time()
            
            # Provide feedback on processing time for manual captures
            if is_capture:
                confidence_msg = f"Text detected with {best_confidence:.1f}% confidence"
                if processing_time > 2:
                    self.speech_manager.say(f"{confidence_msg} in {processing_time:.1f} seconds", priority=True)
                else:
                    self.speech_manager.say(confidence_msg, priority=True)
            
            # Speak the detected text with natural pauses between sentences
            self._speak_text_with_pauses(best_result, priority=priority)
            
            return best_result
            
        except Exception as e:
            logging.error(f"❌ Error in text recognition: {str(e)}")
            if is_capture:
                self.speech_manager.say("Error processing text in image", priority=True)
            return None
        finally:
            self.is_processing = False
            self.should_stop = False

    def _simple_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Simple preprocessing as fallback method.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Resize if too small
            h, w = gray.shape
            if min(h, w) < 100:
                scale = 100 / min(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Basic contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Light denoising
            denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
            
            return denoised
            
        except Exception as e:
            logging.error(f"❌ Error in simple preprocessing: {str(e)}")
            # Return original grayscale as last resort
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image

    def _preprocess_image_advanced(self, image: np.ndarray, strategy: str) -> np.ndarray:
        """
        Advanced image preprocessing with multiple strategies for optimal text recognition.
        
        Args:
            image: Input image
            strategy: Preprocessing strategy to use
            
        Returns:
            Preprocessed image
        """
        try:
            # Create a copy to avoid modifying the original
            processed = image.copy()
            
            # Resize if image is too large (preserves aspect ratio)
            processed = self._resize_image_optimal(processed)
            
            # Apply strategy-specific preprocessing
            if strategy == 'enhanced_default':
                processed = self._enhanced_default_preprocessing(processed)
            elif strategy == 'high_contrast':
                processed = self._high_contrast_preprocessing(processed)
            elif strategy == 'adaptive_threshold':
                processed = self._adaptive_threshold_preprocessing(processed)
            elif strategy == 'edge_enhanced':
                processed = self._edge_enhanced_preprocessing(processed)
            elif strategy == 'denoised':
                processed = self._denoised_preprocessing(processed)
            elif strategy == 'gamma_corrected':
                processed = self._gamma_corrected_preprocessing(processed)
            elif strategy == 'morphological_enhanced':
                processed = self._morphological_enhanced_preprocessing(processed)
            else:
                processed = self._enhanced_default_preprocessing(processed)
                
            return processed
            
        except Exception as e:
            logging.error(f"❌ Error in advanced preprocessing ({strategy}): {str(e)}")
            return self._simple_preprocessing(image)  # Fallback to simple preprocessing

    def _resize_image_optimal(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to optimal dimensions for OCR processing.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        
        # Target dimensions for optimal OCR performance
        min_dimension = 300   # Reduced minimum dimension
        max_dimension = 1500  # Reduced maximum dimension
        
        # Calculate scale factor
        scale = 1.0
        if min(h, w) < min_dimension:
            scale = min_dimension / min(h, w)
        elif max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
        
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            # Use INTER_CUBIC for upscaling, INTER_AREA for downscaling
            interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        return image

    def _enhanced_default_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Enhanced default preprocessing with multiple techniques."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply bilateral filter for noise reduction while preserving edges
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Apply unsharp masking for better text definition
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
            unsharp = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
            
            # Apply gamma correction for better contrast
            gamma = 1.2
            lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                    for i in np.arange(0, 256)]).astype("uint8")
            corrected = cv2.LUT(unsharp, lookup_table)
            
            return corrected
        except Exception as e:
            logging.error(f"❌ Error in enhanced default preprocessing: {str(e)}")
            return self._simple_preprocessing(image)

    def _high_contrast_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """High contrast preprocessing for faded or low contrast text."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(gray)
            
            # Apply adaptive histogram equalization with very high clip limit
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(equalized)
            
            # Apply morphological closing to connect broken text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            # Apply aggressive contrast stretching
            min_val, max_val = np.percentile(closed, [1, 99])
            if max_val > min_val:  # Avoid division by zero
                stretched = np.clip((closed - min_val) * 255 / (max_val - min_val), 0, 255).astype(np.uint8)
            else:
                stretched = closed
            
            return stretched
        except Exception as e:
            logging.error(f"❌ Error in high contrast preprocessing: {str(e)}")
            return self._simple_preprocessing(image)

    def _adaptive_threshold_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Adaptive thresholding for varying lighting conditions."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive threshold
            adaptive_thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
        except Exception as e:
            logging.error(f"❌ Error in adaptive threshold preprocessing: {str(e)}")
            return self._simple_preprocessing(image)

    def _edge_enhanced_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Edge enhancement for better text boundary definition."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply Laplacian edge detection
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            
            # Combine original with edge-enhanced version
            enhanced = cv2.addWeighted(gray, 0.7, laplacian, 0.3, 0)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            result = clahe.apply(enhanced)
            
            return result
        except Exception as e:
            logging.error(f"❌ Error in edge enhanced preprocessing: {str(e)}")
            return self._simple_preprocessing(image)

    def _denoised_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Heavy denoising for very noisy images."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply Non-local Means Denoising
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Apply bilateral filter for additional smoothing
            bilateral = cv2.bilateralFilter(denoised, 9, 80, 80)
            
            # Apply median filter to remove salt and pepper noise
            median = cv2.medianBlur(bilateral, 3)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(median)
            
            return enhanced
        except Exception as e:
            logging.error(f"❌ Error in denoised preprocessing: {str(e)}")
            return self._simple_preprocessing(image)

    def _gamma_corrected_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Gamma correction for exposure issues."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Calculate optimal gamma based on image brightness
            mean_brightness = np.mean(gray)
            if mean_brightness < 100:
                gamma = 0.7  # Brighten dark images
            elif mean_brightness > 180:
                gamma = 1.3  # Darken bright images
            else:
                gamma = 1.0  # No correction needed
            
            # Apply gamma correction
            lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                    for i in np.arange(0, 256)]).astype("uint8")
            corrected = cv2.LUT(gray, lookup_table)
            
            # Apply CLAHE for local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(corrected)
            
            return enhanced
        except Exception as e:
            logging.error(f"❌ Error in gamma corrected preprocessing: {str(e)}")
            return self._simple_preprocessing(image)

    def _morphological_enhanced_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Morphological operations for text structure enhancement."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply initial denoising
            denoised = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Apply morphological operations to enhance text structure
            # Horizontal kernel for connecting letters
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            horizontal_enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, horizontal_kernel)
            
            # Vertical kernel for connecting parts of letters
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            vertical_enhanced = cv2.morphologyEx(horizontal_enhanced, cv2.MORPH_CLOSE, vertical_kernel)
            
            # Apply opening to remove small noise
            noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            cleaned = cv2.morphologyEx(vertical_enhanced, cv2.MORPH_OPEN, noise_kernel)
            
            return cleaned
        except Exception as e:
            logging.error(f"❌ Error in morphological enhanced preprocessing: {str(e)}")
            return self._simple_preprocessing(image)

    def _process_ocr_results_with_confidence(self, results: List) -> Tuple[str, float]:
        """
        Process OCR results and return text with average confidence.
        
        Args:
            results: PaddleOCR detection results
            
        Returns:
            Tuple of (processed text, average confidence)
        """
        if not results:
            return "", 0.0
            
        text_blocks = []
        confidences = []
        
        # Extract text and confidence from results
        for line in results:
            if len(line) >= 2:
                text_info = line[1]
                if len(text_info) >= 2:
                    text = text_info[0]
                    confidence = text_info[1]
                    
                    # Filter out low confidence detections (lowered threshold)
                    if confidence > 0.3 and self._is_valid_text(text):
                        text_blocks.append(text)
                        confidences.append(confidence)
        
        if not text_blocks:
            return "", 0.0
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences)
        
        # Join text blocks with appropriate spacing
        full_text = " ".join(text_blocks)
        
        # Post-process the text
        cleaned_text = self._post_process_text(full_text)
        
        return cleaned_text, avg_confidence

    def _post_process_text(self, text: str) -> str:
        """
        Enhanced post-processing of recognized text.
        
        Args:
            text: Raw recognized text
            
        Returns:
            Processed text
        """
        if not text:
            return ""
            
        # Replace multiple spaces with single space
        processed = " ".join(text.split())
        
        # Enhanced OCR error corrections
        corrections = {
            'l<': 'k',
            'rn': 'm',
            'vv': 'w',
            'VV': 'W',
            'cl': 'd',
            'cl0': 'do',
            'l0': 'lo',
            'O0': '00',
            'S5': 'SS',
            'G6': 'G6',
            'B8': 'B8',
            'I1': 'I1',
            'Z2': 'Z2',
            '|': 'l',
            '1l': 'll',
        }
        
        # Apply corrections
        for wrong, correct in corrections.items():
            processed = processed.replace(wrong, correct)
        
        # Fix sentence spacing
        for punct in ".!?":
            processed = processed.replace(f"{punct} ", f"{punct} ")
            processed = processed.replace(f"{punct}", f"{punct} ")
        
        # Remove excessive punctuation
        processed = processed.replace(",,", ",")
        processed = processed.replace("..", ".")
        processed = processed.replace("!!", "!")
        processed = processed.replace("??", "?")
        
        # Fix common word boundaries
        processed = processed.replace(" ,", ",")
        processed = processed.replace(" .", ".")
        processed = processed.replace(" !", "!")
        processed = processed.replace(" ?", "?")
        
        return processed.strip()

    def _is_valid_text(self, text: str) -> bool:
        """
        Enhanced validation for detected text (with relaxed criteria).
        
        Args:
            text: Text to validate
            
        Returns:
            Whether text is valid
        """
        if not text or len(text.strip()) == 0:
            return False
            
        # Text should contain some letters (relaxed requirement)
        alpha_count = sum(c.isalpha() for c in text)
        if alpha_count < 1:
            return False
            
        # Check for minimum meaningful length (relaxed)
        if len(text.strip()) < 1:
            return False
            
        # Check that the text has a reasonable ratio of alphanumeric characters (relaxed)
        alphanum_ratio = sum(c.isalnum() for c in text) / len(text)
        if alphanum_ratio < 0.2:  # At least 20% should be alphanumeric (was 40%)
            return False
        
        # Check for too many repeated characters (relaxed)
        if len(text) > 3 and len(set(text)) < len(text) * 0.2:  # Less than 20% unique characters (was 30%)
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
