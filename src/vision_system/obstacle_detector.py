
"""
obstacle_detector.py - Enhanced Obstacle Detection Module for Smart White Glass
Designed for parallel execution with computer vision modules
Author: Smart White Glass Project
"""

import RPi.GPIO as GPIO
import time
import threading
import queue
import logging
from typing import Optional, Callable, Dict, Any
from enum import Enum
from contextlib import contextmanager
import json
from dataclasses import dataclass, asdict

# === Configuration ===
class AlertLevel(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

@dataclass
class ObstacleConfig:
    """Configuration class for obstacle detector"""
    # GPIO Pins
    trig_pin: int = 23
    echo_pin: int = 24
    led1_pin: int = 17
    led2_pin: int = 27
    vibration_pin: int = 18
    
    # Distance thresholds (in cm)
    far_threshold: int = 100
    near_threshold: int = 50
    critical_threshold: int = 20
    
    # Timing settings
    measurement_interval: float = 0.1  # 100ms between measurements
    max_measurement_time: float = 0.05  # Timeout for ultrasonic measurement
    
    # PWM settings
    vibration_frequency: int = 100
    low_vibration_duty: int = 30
    medium_vibration_duty: int = 50
    high_vibration_duty: int = 80
    
    # Performance settings
    max_queue_size: int = 10
    enable_logging: bool = True
    enable_callbacks: bool = True

@dataclass
class ObstacleData:
    """Data structure for obstacle detection results"""
    distance: float
    alert_level: AlertLevel
    timestamp: float
    is_valid: bool = True
    error_message: Optional[str] = None

class ObstacleDetector:
    """
    Enhanced Obstacle Detector for Smart White Glass
    Designed for thread-safe operation with other CV modules
    """
    
    def __init__(self, config: ObstacleConfig = None):
        self.config = config or ObstacleConfig()
        self._running = False
        self._thread = None
        self._data_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self._callbacks = []
        self._lock = threading.Lock()
        self._gpio_initialized = False
        self._vibration_pwm = None
        
        # Setup logging
        if self.config.enable_logging:
            self._setup_logging()
        
        # Initialize GPIO
        self._init_gpio()
    
    def _setup_logging(self):
        """Setup logging for the obstacle detector"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ObstacleDetector - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_gpio(self):
        """Initialize GPIO pins safely"""
        try:
            if not self._gpio_initialized:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                
                # Setup pins
                GPIO.setup(self.config.trig_pin, GPIO.OUT)
                GPIO.setup(self.config.echo_pin, GPIO.IN)
                GPIO.setup(self.config.led1_pin, GPIO.OUT)
                GPIO.setup(self.config.led2_pin, GPIO.OUT)
                GPIO.setup(self.config.vibration_pin, GPIO.OUT)
                
                # Initialize PWM for vibration motor
                self._vibration_pwm = GPIO.PWM(self.config.vibration_pin, self.config.vibration_frequency)
                self._vibration_pwm.start(0)
                
                # Reset all outputs
                self._reset_outputs()
                
                self._gpio_initialized = True
                if hasattr(self, 'logger'):
                    self.logger.info("GPIO initialized successfully")
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"GPIO initialization failed: {e}")
            raise
    
    def _reset_outputs(self):
        """Reset all output devices to safe state"""
        try:
            GPIO.output(self.config.led1_pin, False)
            GPIO.output(self.config.led2_pin, False)
            if self._vibration_pwm:
                self._vibration_pwm.ChangeDutyCycle(0)
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error resetting outputs: {e}")
    
    def _measure_distance(self) -> Optional[float]:
        """
        Measure distance using ultrasonic sensor with timeout protection
        Returns distance in cm or None if measurement fails
        """
        try:
            # Send trigger pulse
            GPIO.output(self.config.trig_pin, True)
            time.sleep(0.00001)  # 10 microseconds
            GPIO.output(self.config.trig_pin, False)
            
            # Wait for echo start with timeout
            start_time = time.time()
            timeout_start = start_time + self.config.max_measurement_time
            
            while GPIO.input(self.config.echo_pin) == 0:
                start_time = time.time()
                if start_time > timeout_start:
                    return None  # Timeout
            
            # Wait for echo end with timeout
            stop_time = time.time()
            timeout_end = stop_time + self.config.max_measurement_time
            
            while GPIO.input(self.config.echo_pin) == 1:
                stop_time = time.time()
                if stop_time > timeout_end:
                    return None  # Timeout
            
            # Calculate distance
            elapsed = stop_time - start_time
            distance = (elapsed * 34300) / 2  # Convert to cm
            
            # Validate distance (reasonable range)
            if 0 < distance <= 400:  # Max range for typical HC-SR04
                return distance
            else:
                return None
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Distance measurement error: {e}")
            return None
    
    def _determine_alert_level(self, distance: float) -> AlertLevel:
        """Determine alert level based on distance"""
        if distance <= self.config.critical_threshold:
            return AlertLevel.HIGH
        elif distance <= self.config.near_threshold:
            return AlertLevel.MEDIUM
        elif distance <= self.config.far_threshold:
            return AlertLevel.LOW
        else:
            return AlertLevel.NONE
    
    def _update_outputs(self, alert_level: AlertLevel):
        """Update LED and vibration outputs based on alert level"""
        try:
            # Reset outputs first
            self._reset_outputs()
            
            if alert_level == AlertLevel.LOW:
                GPIO.output(self.config.led1_pin, True)
                self._vibration_pwm.ChangeDutyCycle(self.config.low_vibration_duty)
                
            elif alert_level == AlertLevel.MEDIUM:
                GPIO.output(self.config.led1_pin, True)
                self._vibration_pwm.ChangeDutyCycle(self.config.medium_vibration_duty)
                
            elif alert_level == AlertLevel.HIGH:
                GPIO.output(self.config.led2_pin, True)
                self._vibration_pwm.ChangeDutyCycle(self.config.high_vibration_duty)
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error updating outputs: {e}")
    
    def _detection_loop(self):
        """Main detection loop running in separate thread"""
        if hasattr(self, 'logger'):
            self.logger.info("Obstacle detection started")
        
        while self._running:
            try:
                # Measure distance
                distance = self._measure_distance()
                timestamp = time.time()
                
                if distance is not None:
                    # Determine alert level
                    alert_level = self._determine_alert_level(distance)
                    
                    # Update outputs
                    self._update_outputs(alert_level)
                    
                    # Create obstacle data
                    obstacle_data = ObstacleData(
                        distance=distance,
                        alert_level=alert_level,
                        timestamp=timestamp,
                        is_valid=True
                    )
                    
                else:
                    # Invalid measurement
                    obstacle_data = ObstacleData(
                        distance=-1,
                        alert_level=AlertLevel.NONE,
                        timestamp=timestamp,
                        is_valid=False,
                        error_message="Measurement timeout or invalid reading"
                    )
                    self._reset_outputs()
                
                # Add to queue (non-blocking)
                try:
                    self._data_queue.put_nowait(obstacle_data)
                except queue.Full:
                    # Remove oldest item and add new one
                    try:
                        self._data_queue.get_nowait()
                        self._data_queue.put_nowait(obstacle_data)
                    except queue.Empty:
                        pass
                
                # Call registered callbacks
                if self.config.enable_callbacks:
                    self._call_callbacks(obstacle_data)
                
                # Sleep between measurements
                time.sleep(self.config.measurement_interval)
                
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Error in detection loop: {e}")
                self._reset_outputs()
                time.sleep(self.config.measurement_interval)
    
    def _call_callbacks(self, data: ObstacleData):
        """Call all registered callbacks with obstacle data"""
        with self._lock:
            for callback in self._callbacks:
                try:
                    callback(data)
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"Callback error: {e}")
    
    def start(self):
        """Start obstacle detection in separate thread"""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._detection_loop, daemon=True)
            self._thread.start()
            if hasattr(self, 'logger'):
                self.logger.info("Obstacle detector started")
    
    def stop(self):
        """Stop obstacle detection"""
        if self._running:
            self._running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
            self._reset_outputs()
            if hasattr(self, 'logger'):
                self.logger.info("Obstacle detector stopped")
    
    def get_latest_data(self) -> Optional[ObstacleData]:
        """Get the most recent obstacle detection data"""
        try:
            return self._data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_data(self) -> list:
        """Get all available obstacle detection data"""
        data_list = []
        while True:
            try:
                data_list.append(self._data_queue.get_nowait())
            except queue.Empty:
                break
        return data_list
    
    def register_callback(self, callback: Callable[[ObstacleData], None]):
        """Register a callback function to be called with new obstacle data"""
        with self._lock:
            self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[ObstacleData], None]):
        """Unregister a callback function"""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def is_running(self) -> bool:
        """Check if obstacle detection is running"""
        return self._running
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the obstacle detector"""
        return {
            'running': self._running,
            'gpio_initialized': self._gpio_initialized,
            'queue_size': self._data_queue.qsize(),
            'callback_count': len(self._callbacks) if hasattr(self, '_callbacks') else 0,
            'config': asdict(self.config)
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.stop()
        try:
            if self._vibration_pwm:
                self._vibration_pwm.ChangeDutyCycle(0)
                self._vibration_pwm.stop()
            if self._gpio_initialized:
                GPIO.cleanup()
            if hasattr(self, 'logger'):
                self.logger.info("Cleanup completed")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Cleanup error: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

# === Convenience Functions ===

def create_obstacle_detector(config_dict: Dict = None) -> ObstacleDetector:
    """Create obstacle detector with optional configuration dictionary"""
    if config_dict:
        config = ObstacleConfig(**config_dict)
    else:
        config = ObstacleConfig()
    return ObstacleDetector(config)

# === Example Usage ===
if __name__ == "__main__":
    # Example callback function
    def obstacle_callback(data: ObstacleData):
        if data.is_valid:
            print(f"Distance: {data.distance:.2f} cm, Alert: {data.alert_level.name}")
        else:
            print(f"Invalid reading: {data.error_message}")
    
    # Create custom configuration
    config = ObstacleConfig(
        measurement_interval=0.05,  # 50ms for faster response
        near_threshold=40,          # Closer threshold
        critical_threshold=15       # More critical threshold
    )
    
    # Use context manager for automatic cleanup
    try:
        with ObstacleDetector(config) as detector:
            # Register callback
            detector.register_callback(obstacle_callback)
            
            # Let it run for 30 seconds
            print("Obstacle detection running... Press Ctrl+C to stop")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\nStopping obstacle detection...")
    except Exception as e:
        print(f"Error: {e}")