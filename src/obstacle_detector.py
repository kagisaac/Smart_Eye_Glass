
"""
Obstacle Detector Module for Smart Eye Glass System
Provides ultrasonic sensor-based obstacle detection with audio alerts.
"""

import time
import threading
import queue
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, List
import logging

# For Raspberry Pi GPIO (you'll need to install RPi.GPIO)
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("⚠️ RPi.GPIO not available - running in simulation mode")

class AlertLevel(Enum):
    """Alert levels for obstacle detection."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

@dataclass
class ObstacleData:
    """Data structure for obstacle detection results."""
    distance: float
    alert_level: AlertLevel
    timestamp: float
    is_valid: bool = True
    sensor_id: str = "primary"
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.distance < 0:
            self.is_valid = False
        if self.timestamp <= 0:
            self.timestamp = time.time()

@dataclass
class ObstacleConfig:
    """Configuration for obstacle detection system."""
    measurement_interval: float = 0.1  # seconds between measurements
    near_threshold: float = 60.0       # cm - warning distance
    critical_threshold: float = 25.0   # cm - critical distance
    far_threshold: float = 100.0       # cm - initial detection distance
    max_distance: float = 400.0        # cm - sensor max range
    trigger_pin: int = 18              # GPIO pin for trigger
    echo_pin: int = 24                 # GPIO pin for echo
    enable_logging: bool = False       # Enable detailed logging
    enable_callbacks: bool = True      # Enable callback system
    smoothing_window: int = 3          # Number of readings to average
    timeout_duration: float = 0.04     # Timeout for echo response (40ms)

class ObstacleDetector:
    """
    Ultrasonic sensor-based obstacle detection system.
    Provides real-time distance measurements and alert levels.
    """
    
    def __init__(self, config: ObstacleConfig):
        self.config = config
        self.running = False
        self.measurement_thread = None
        self.callbacks: List[Callable[[ObstacleData], None]] = []
        self.last_readings = queue.Queue(maxsize=config.smoothing_window)
        self.last_alert_time = {}  # Track last alert time for each level
        self.detection_active = True
        
        # Setup logging if enabled
        if config.enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
        
        # Initialize GPIO if available
        if GPIO_AVAILABLE:
            self._setup_gpio()
        else:
            self._log("GPIO not available - using simulation mode")
    
    def _setup_gpio(self):
        """Setup GPIO pins for ultrasonic sensor."""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.config.trigger_pin, GPIO.OUT)
            GPIO.setup(self.config.echo_pin, GPIO.IN)
            GPIO.output(self.config.trigger_pin, False)
            time.sleep(0.1)  # Let sensor settle
            self._log("GPIO setup complete")
        except Exception as e:
            self._log(f"GPIO setup failed: {e}")
            raise
    
    def _log(self, message: str):
        """Log message if logging is enabled."""
        if self.logger:
            self.logger.info(message)
        else:
            print(f"[ObstacleDetector] {message}")
    
    def start(self):
        """Start the obstacle detection system."""
        if self.running:
            self._log("Already running")
            return
        
        self.running = True
        self.measurement_thread = threading.Thread(target=self._measurement_loop, daemon=True)
        self.measurement_thread.start()
        self._log("Obstacle detection started")
    
    def stop(self):
        """Stop the obstacle detection system."""
        self.running = False
        if self.measurement_thread and self.measurement_thread.is_alive():
            self.measurement_thread.join(timeout=2.0)
        self._log("Obstacle detection stopped")
    
    def _measurement_loop(self):
        """Main measurement loop running in separate thread."""
        while self.running:
            try:
                if self.detection_active:
                    distance = self._measure_distance()
                    if distance is not None:
                        obstacle_data = self._process_measurement(distance)
                        if obstacle_data and self.config.enable_callbacks:
                            self._trigger_callbacks(obstacle_data)
                
                time.sleep(self.config.measurement_interval)
                
            except Exception as e:
                self._log(f"Error in measurement loop: {e}")
                time.sleep(1.0)  # Wait before retrying
    
    def _measure_distance(self) -> Optional[float]:
        """
        Measure distance using ultrasonic sensor.
        Returns distance in centimeters or None if measurement failed.
        """
        if not GPIO_AVAILABLE:
            # Simulation mode - return random distance for testing
            import random
            return random.uniform(10, 200)
        
        try:
            # Send trigger pulse
            GPIO.output(self.config.trigger_pin, True)
            time.sleep(0.00001)  # 10µs pulse
            GPIO.output(self.config.trigger_pin, False)
            
            # Wait for echo start
            pulse_start = time.time()
            timeout_start = pulse_start
            
            while GPIO.input(self.config.echo_pin) == 0:
                pulse_start = time.time()
                if pulse_start - timeout_start > self.config.timeout_duration:
                    self._log("Timeout waiting for echo start")
                    return None
            
            # Wait for echo end
            pulse_end = time.time()
            timeout_end = pulse_end
            
            while GPIO.input(self.config.echo_pin) == 1:
                pulse_end = time.time()
                if pulse_end - timeout_end > self.config.timeout_duration:
                    self._log("Timeout waiting for echo end")
                    return None
            
            # Calculate distance
            pulse_duration = pulse_end - pulse_start
            distance = (pulse_duration * 34300) / 2  # Speed of sound: 343 m/s
            
            # Validate measurement
            if 0 < distance <= self.config.max_distance:
                return distance
            else:
                return None
                
        except Exception as e:
            self._log(f"Distance measurement error: {e}")
            return None
    
    def _process_measurement(self, distance: float) -> Optional[ObstacleData]:
        """Process distance measurement and determine alert level."""
        # Add to smoothing window
        if self.last_readings.full():
            try:
                self.last_readings.get_nowait()
            except queue.Empty:
                pass
        
        self.last_readings.put(distance)
        
        # Calculate smoothed distance
        readings = []
        temp_queue = queue.Queue()
        
        while not self.last_readings.empty():
            try:
                reading = self.last_readings.get_nowait()
                readings.append(reading)
                temp_queue.put(reading)
            except queue.Empty:
                break
        
        # Restore queue
        while not temp_queue.empty():
            try:
                self.last_readings.put_nowait(temp_queue.get_nowait())
            except queue.Full:
                break
        
        if not readings:
            return None
        
        smoothed_distance = sum(readings) / len(readings)
        
        # Determine alert level
        alert_level = self._calculate_alert_level(smoothed_distance)
        
        # Create obstacle data
        obstacle_data = ObstacleData(
            distance=smoothed_distance,
            alert_level=alert_level,
            timestamp=time.time(),
            is_valid=True
        )
        
        return obstacle_data
    
    def _calculate_alert_level(self, distance: float) -> AlertLevel:
        """Calculate alert level based on distance."""
        if distance <= self.config.critical_threshold:
            return AlertLevel.HIGH
        elif distance <= self.config.near_threshold:
            return AlertLevel.MEDIUM
        elif distance <= self.config.far_threshold:
            return AlertLevel.LOW
        else:
            return AlertLevel.NONE
    
    def _trigger_callbacks(self, obstacle_data: ObstacleData):
        """Trigger registered callbacks with obstacle data."""
        # Implement rate limiting for alerts
        current_time = time.time()
        alert_level = obstacle_data.alert_level
        
        # Rate limiting based on alert level
        if alert_level == AlertLevel.HIGH:
            min_interval = 0.5  # Critical alerts every 0.5 seconds
        elif alert_level == AlertLevel.MEDIUM:
            min_interval = 1.0  # Warning alerts every 1 second
        elif alert_level == AlertLevel.LOW:
            min_interval = 2.0  # Low alerts every 2 seconds
        else:
            min_interval = 5.0  # Info alerts every 5 seconds
        
        last_time = self.last_alert_time.get(alert_level, 0)
        
        if current_time - last_time >= min_interval:
            self.last_alert_time[alert_level] = current_time
            
            # Call all registered callbacks
            for callback in self.callbacks:
                try:
                    callback(obstacle_data)
                except Exception as e:
                    self._log(f"Callback error: {e}")
    
    def register_callback(self, callback: Callable[[ObstacleData], None]):
        """Register a callback function to receive obstacle data."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
            self._log(f"Callback registered: {callback.__name__}")
    
    def unregister_callback(self, callback: Callable[[ObstacleData], None]):
        """Unregister a callback function."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            self._log(f"Callback unregistered: {callback.__name__}")
    
    def get_current_distance(self) -> Optional[float]:
        """Get current distance measurement synchronously."""
        return self._measure_distance()
    
    def set_detection_active(self, active: bool):
        """Enable or disable detection measurements."""
        self.detection_active = active
        self._log(f"Detection active: {active}")
    
    def get_status(self) -> dict:
        """Get current status of the obstacle detector."""
        return {
            'running': self.running,
            'detection_active': self.detection_active,
            'gpio_available': GPIO_AVAILABLE,
            'config': {
                'measurement_interval': self.config.measurement_interval,
                'near_threshold': self.config.near_threshold,
                'critical_threshold': self.config.critical_threshold,
                'far_threshold': self.config.far_threshold,
                'trigger_pin': self.config.trigger_pin,
                'echo_pin': self.config.echo_pin
            },
            'callbacks_registered': len(self.callbacks),
            'smoothing_window': self.config.smoothing_window
        }
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self._log(f"Config updated: {key} = {value}")
    
    def cleanup(self):
        """Clean up resources and stop detection."""
        self._log("Cleaning up obstacle detector...")
        self.stop()
        
        if GPIO_AVAILABLE:
            try:
                GPIO.cleanup([self.config.trigger_pin, self.config.echo_pin])
                self._log("GPIO cleanup complete")
            except Exception as e:
                self._log(f"GPIO cleanup error: {e}")

# Test and utility functions
def test_obstacle_detector():
    """Test function for obstacle detector."""
    print("🧪 Testing Obstacle Detector...")
    
    config = ObstacleConfig(
        measurement_interval=0.5,
        near_threshold=50,
        critical_threshold=20,
        enable_logging=True 
    )
    
    detector = ObstacleDetector(config)
    
    def test_callback(data: ObstacleData):
        print(f"📏 Distance: {data.distance:.1f}cm, Alert: {data.alert_level.name}")
    
    detector.register_callback(test_callback)
    
    try:
        detector.start()
        print("🔄 Running for 30 seconds... Press Ctrl+C to stop")
        time.sleep(30)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
    finally:
        detector.cleanup()
        print("✅ Test complete")

if __name__ == "__main__":
    test_obstacle_detector()

