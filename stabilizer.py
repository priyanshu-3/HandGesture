"""
Temporal Stabilization Module
Smooths landmarks and reduces jitter using filters
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from collections import deque
from typing import Optional, List, Dict


class LandmarkStabilizer:
    """
    Stabilizes landmark positions using temporal filtering
    """
    
    def __init__(self, num_landmarks: int = 21, smoothing: float = 0.5):
        """
        Initialize stabilizer
        
        Args:
            num_landmarks: Number of landmarks to stabilize
            smoothing: Smoothing factor (0 = no smoothing, 1 = max smoothing)
        """
        self.num_landmarks = num_landmarks
        self.smoothing = np.clip(smoothing, 0.0, 0.95)
        
        # Exponential moving average for each landmark
        self.ema_landmarks = None
        
        # Velocity estimation for prediction
        self.prev_landmarks = None
        self.velocity = None
        
        # Outlier detection
        self.history_length = 5
        self.landmark_history = deque(maxlen=self.history_length)
        
    def update(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Update and smooth landmarks
        
        Args:
            landmarks: Current landmarks (N x 3)
            
        Returns:
            Smoothed landmarks
        """
        if landmarks is None or len(landmarks) == 0:
            return landmarks
        
        # Initialize on first frame
        if self.ema_landmarks is None:
            self.ema_landmarks = landmarks.copy()
            self.prev_landmarks = landmarks.copy()
            self.velocity = np.zeros_like(landmarks)
            return landmarks
        
        # Outlier detection
        if self._is_outlier(landmarks):
            # Use prediction instead
            predicted = self.ema_landmarks + self.velocity
            return predicted
        
        # Compute velocity
        self.velocity = landmarks - self.prev_landmarks
        
        # Exponential moving average
        self.ema_landmarks = (self.smoothing * self.ema_landmarks + 
                             (1 - self.smoothing) * landmarks)
        
        # Update history
        self.prev_landmarks = landmarks.copy()
        self.landmark_history.append(landmarks)
        
        return self.ema_landmarks
    
    def _is_outlier(self, landmarks: np.ndarray, threshold: float = 0.15) -> bool:
        """
        Detect if landmarks are outliers (sudden jump)
        
        Args:
            landmarks: Current landmarks
            threshold: Maximum allowed movement (as fraction of typical movement)
            
        Returns:
            True if outlier detected
        """
        if self.prev_landmarks is None or len(self.landmark_history) < 2:
            return False
        
        # Compute movement
        movement = np.linalg.norm(landmarks - self.prev_landmarks, axis=1)
        max_movement = np.max(movement)
        
        # Compute typical movement from history
        if len(self.landmark_history) >= 2:
            typical_movements = []
            for i in range(1, len(self.landmark_history)):
                hist_movement = np.linalg.norm(
                    self.landmark_history[i] - self.landmark_history[i-1],
                    axis=1
                )
                typical_movements.append(np.max(hist_movement))
            
            avg_movement = np.mean(typical_movements)
            
            # If current movement is much larger than typical, it's an outlier
            if max_movement > avg_movement * 3 and max_movement > threshold:
                return True
        
        return False
    
    def reset(self):
        """Reset stabilizer state"""
        self.ema_landmarks = None
        self.prev_landmarks = None
        self.velocity = None
        self.landmark_history.clear()


class OneEuroFilter:
    """
    One Euro Filter for low-latency smoothing
    Better than simple EMA for interactive applications
    """
    
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007):
        """
        Initialize One Euro Filter
        
        Args:
            min_cutoff: Minimum cutoff frequency
            beta: Speed coefficient
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = 1.0
        
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
        
    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Filter value
        
        Args:
            x: Current value (can be multi-dimensional)
            t: Current timestamp
            
        Returns:
            Filtered value
        """
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x
        
        # Time step
        dt = t - self.t_prev
        if dt <= 0:
            dt = 0.001
        
        # Estimate velocity
        dx = (x - self.x_prev) / dt
        
        # Smooth velocity
        alpha_d = self._smoothing_factor(dt, self.d_cutoff)
        dx_hat = self._exponential_smoothing(alpha_d, dx, self.dx_prev)
        
        # Adaptive cutoff based on velocity
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        
        # Smooth position
        alpha = self._smoothing_factor(dt, cutoff)
        x_hat = self._exponential_smoothing(alpha, x, self.x_prev)
        
        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat
    
    def _smoothing_factor(self, dt: float, cutoff: float) -> float:
        """Calculate smoothing factor"""
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)
    
    def _exponential_smoothing(self, alpha: float, x: np.ndarray, 
                               x_prev: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing"""
        return alpha * x + (1 - alpha) * x_prev


class VelocityStabilizer:
    """
    Stabilizes velocities and accelerations for smooth interactions
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize velocity stabilizer
        
        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
        self.position_history = deque(maxlen=window_size)
        self.time_history = deque(maxlen=window_size)
        
    def update(self, position: np.ndarray, timestamp: float) -> Dict:
        """
        Update and compute smoothed velocity
        
        Args:
            position: Current position (3D)
            timestamp: Current time
            
        Returns:
            Dict with velocity and acceleration
        """
        self.position_history.append(position)
        self.time_history.append(timestamp)
        
        if len(self.position_history) < 2:
            return {
                'velocity': np.zeros(3),
                'speed': 0.0,
                'acceleration': np.zeros(3)
            }
        
        # Compute velocity using linear regression over window
        positions = np.array(self.position_history)
        times = np.array(self.time_history)
        times = times - times[0]  # Normalize
        
        # Simple finite difference for velocity
        dt = times[-1] - times[-2]
        if dt > 0:
            velocity = (positions[-1] - positions[-2]) / dt
        else:
            velocity = np.zeros(3)
        
        speed = np.linalg.norm(velocity)
        
        # Acceleration (if enough history)
        acceleration = np.zeros(3)
        if len(self.position_history) >= 3:
            dt2 = times[-1] - times[-3]
            if dt2 > 0:
                v_prev = (positions[-2] - positions[-3]) / (times[-2] - times[-3])
                acceleration = (velocity - v_prev) / dt
        
        return {
            'velocity': velocity,
            'speed': speed,
            'acceleration': acceleration
        }

