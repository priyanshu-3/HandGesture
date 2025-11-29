"""
Gesture Recognition Module
Recognizes discrete and continuous gestures from 3D hand models
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from hand_model_3d import HandModel3D
from collections import deque


class GestureRecognizer:
    """
    Recognizes hand gestures for interaction
    """
    
    # Gesture definitions
    GESTURES = {
        'PINCH': 'pinch',
        'GRAB': 'grab',
        'POINT': 'point',
        'OPEN_PALM': 'open_palm',
        'FIST': 'fist',
        'PEACE': 'peace',
        'OK': 'ok',
        'THUMBS_UP': 'thumbs_up',
        'NONE': 'none'
    }
    
    def __init__(self, history_length: int = 10):
        """
        Initialize gesture recognizer
        
        Args:
            history_length: Number of frames to keep for temporal gestures
        """
        self.history_length = history_length
        self.gesture_history = {
            'left': deque(maxlen=history_length),
            'right': deque(maxlen=history_length)
        }
        
        # Gesture state machines
        self.gesture_states = {
            'left': {'current': 'none', 'confidence': 0.0, 'frames': 0},
            'right': {'current': 'none', 'confidence': 0.0, 'frames': 0}
        }
        
        # Thresholds
        self.pinch_distance_threshold = 0.04  # meters
        self.grab_curl_threshold = 0.7
        self.min_confirmation_frames = 3
        
    def recognize(self, hand_model: HandModel3D, hand_label: str) -> Dict:
        """
        Recognize gesture from hand model
        
        Args:
            hand_model: 3D hand model with landmarks
            hand_label: 'Left' or 'Right'
            
        Returns:
            Dictionary with gesture info
        """
        label_key = hand_label.lower()
        
        # Get finger curls
        curls = hand_model.get_all_finger_curls()
        
        # Check various gesture patterns
        gesture, confidence = self._classify_gesture(hand_model, curls)
        
        # Update history
        self.gesture_history[label_key].append(gesture)
        
        # Temporal smoothing and confirmation
        gesture, confidence = self._smooth_gesture(gesture, confidence, label_key)
        
        # Get additional gesture parameters
        params = self._get_gesture_parameters(hand_model, gesture)
        
        return {
            'gesture': gesture,
            'confidence': confidence,
            'parameters': params,
            'curls': curls
        }
    
    def _classify_gesture(self, hand_model: HandModel3D, 
                         curls: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify gesture based on hand configuration
        
        Returns:
            (gesture_name, confidence)
        """
        # Check pinch (thumb-index close)
        pinch_dist = hand_model.get_fingertip_distance('thumb', 'index')
        if pinch_dist < self.pinch_distance_threshold:
            return (self.GESTURES['PINCH'], 0.9)
        
        # Check for OK gesture (thumb-middle close, index extended)
        ok_dist = hand_model.get_fingertip_distance('thumb', 'middle')
        if (ok_dist < self.pinch_distance_threshold and 
            curls['index'] < 0.3):
            return (self.GESTURES['OK'], 0.85)
        
        # Check fist (all fingers curled)
        if all(curl > 0.6 for curl in curls.values()):
            return (self.GESTURES['FIST'], 0.9)
        
        # Check open palm (all fingers extended)
        if all(curl < 0.3 for curl in curls.values()):
            return (self.GESTURES['OPEN_PALM'], 0.9)
        
        # Check point (index extended, others curled)
        if (curls['index'] < 0.3 and 
            curls['middle'] > 0.5 and 
            curls['ring'] > 0.5 and 
            curls['pinky'] > 0.5):
            return (self.GESTURES['POINT'], 0.85)
        
        # Check peace sign (index and middle extended, others curled)
        if (curls['index'] < 0.3 and 
            curls['middle'] < 0.3 and 
            curls['ring'] > 0.5 and 
            curls['pinky'] > 0.5):
            return (self.GESTURES['PEACE'], 0.85)
        
        # Check thumbs up
        if (curls['thumb'] < 0.3 and 
            all(curls[f] > 0.6 for f in ['index', 'middle', 'ring', 'pinky'])):
            return (self.GESTURES['THUMBS_UP'], 0.85)
        
        # Check grab (thumb extended, fingers curled)
        if (curls['thumb'] < 0.4 and 
            curls['index'] > self.grab_curl_threshold and
            curls['middle'] > self.grab_curl_threshold):
            return (self.GESTURES['GRAB'], 0.8)
        
        return (self.GESTURES['NONE'], 0.5)
    
    def _smooth_gesture(self, gesture: str, confidence: float, 
                       label_key: str) -> Tuple[str, float]:
        """
        Apply temporal smoothing to reduce jitter
        
        Args:
            gesture: Current gesture
            confidence: Current confidence
            label_key: 'left' or 'right'
            
        Returns:
            (smoothed_gesture, smoothed_confidence)
        """
        state = self.gesture_states[label_key]
        
        # Check if gesture matches current state
        if gesture == state['current']:
            state['frames'] += 1
            state['confidence'] = min(1.0, state['confidence'] + 0.1)
        else:
            # New gesture detected
            if state['frames'] < self.min_confirmation_frames:
                # Not confirmed yet, keep old gesture
                gesture = state['current']
                confidence = state['confidence']
            else:
                # Switch to new gesture
                state['current'] = gesture
                state['confidence'] = confidence
                state['frames'] = 1
        
        return (gesture, confidence)
    
    def _get_gesture_parameters(self, hand_model: HandModel3D, 
                                gesture: str) -> Dict:
        """
        Get additional parameters for the gesture
        
        Returns:
            Dictionary with gesture-specific parameters
        """
        params = {}
        
        if gesture == self.GESTURES['PINCH']:
            # Pinch strength (based on distance)
            dist = hand_model.get_fingertip_distance('thumb', 'index')
            strength = 1.0 - min(dist / self.pinch_distance_threshold, 1.0)
            params['strength'] = strength
            
            # Pinch position (midpoint between thumb and index)
            thumb_pos = hand_model.get_fingertip_position('thumb')
            index_pos = hand_model.get_fingertip_position('index')
            if thumb_pos is not None and index_pos is not None:
                params['position'] = (thumb_pos + index_pos) / 2
        
        elif gesture == self.GESTURES['POINT']:
            # Point direction (index finger direction)
            index_pos = hand_model.get_fingertip_position('index')
            wrist_pos = hand_model.get_wrist_position()
            if index_pos is not None and wrist_pos is not None:
                direction = index_pos - wrist_pos
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                params['direction'] = direction
                params['position'] = index_pos
        
        elif gesture == self.GESTURES['GRAB']:
            # Grab strength (based on finger curl)
            curls = hand_model.get_all_finger_curls()
            strength = np.mean([curls['index'], curls['middle'], curls['ring']])
            params['strength'] = strength
            
            # Grab position (palm center)
            wrist_pos = hand_model.get_wrist_position()
            if wrist_pos is not None:
                params['position'] = wrist_pos
        
        elif gesture == self.GESTURES['OPEN_PALM']:
            # Palm center position
            wrist_pos = hand_model.get_wrist_position()
            if wrist_pos is not None:
                params['position'] = wrist_pos
            
            # Palm orientation
            orientation = hand_model.get_hand_orientation()
            if 'palm_normal' in orientation:
                params['normal'] = orientation['palm_normal']
        
        return params
    
    def detect_two_hand_gestures(self, left_gesture: Dict, 
                                 right_gesture: Dict) -> Optional[Dict]:
        """
        Detect gestures that require two hands
        
        Args:
            left_gesture: Gesture dict for left hand
            right_gesture: Gesture dict for right hand
            
        Returns:
            Two-hand gesture dict or None
        """
        if left_gesture is None or right_gesture is None:
            return None
        
        left_type = left_gesture['gesture']
        right_type = right_gesture['gesture']
        
        # Two-hand pinch (both hands pinching)
        if (left_type == self.GESTURES['PINCH'] and 
            right_type == self.GESTURES['PINCH']):
            
            left_pos = left_gesture['parameters'].get('position')
            right_pos = right_gesture['parameters'].get('position')
            
            if left_pos is not None and right_pos is not None:
                distance = np.linalg.norm(right_pos - left_pos)
                
                return {
                    'gesture': 'two_hand_pinch',
                    'distance': distance,
                    'left_pos': left_pos,
                    'right_pos': right_pos
                }
        
        # Two-hand open (scaling gesture)
        if (left_type == self.GESTURES['OPEN_PALM'] and 
            right_type == self.GESTURES['OPEN_PALM']):
            
            left_pos = left_gesture['parameters'].get('position')
            right_pos = right_gesture['parameters'].get('position')
            
            if left_pos is not None and right_pos is not None:
                distance = np.linalg.norm(right_pos - left_pos)
                
                return {
                    'gesture': 'two_hand_scale',
                    'distance': distance,
                    'left_pos': left_pos,
                    'right_pos': right_pos
                }
        
        return None
    
    def get_gesture_description(self, gesture: str) -> str:
        """Get human-readable description of gesture"""
        descriptions = {
            'pinch': 'Pinch (Thumb-Index)',
            'grab': 'Grab',
            'point': 'Point',
            'open_palm': 'Open Palm',
            'fist': 'Fist',
            'peace': 'Peace Sign',
            'ok': 'OK Sign',
            'thumbs_up': 'Thumbs Up',
            'none': 'No Gesture'
        }
        return descriptions.get(gesture, gesture)


