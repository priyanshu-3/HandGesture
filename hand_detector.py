"""
Hand Detection Module using MediaPipe
Provides 2D landmark detection with hand tracking
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Optional, Tuple


class HandDetector:
    """Wrapper for MediaPipe Hands with additional utilities"""
    
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize hand detector
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Landmark indices for reference
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        
        # Hand tracking history for consistent labeling
        self.hand_history = []
        self.max_history = 5
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect hands in frame and extract landmarks
        
        Args:
            frame: BGR image from camera
            
        Returns:
            List of hand data dictionaries containing landmarks, label, bbox
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hands_data = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, 
                results.multi_handedness
            ):
                # Extract 2D landmarks (normalized 0-1)
                landmarks_2d = []
                for landmark in hand_landmarks.landmark:
                    landmarks_2d.append([landmark.x, landmark.y, landmark.z])
                
                landmarks_2d = np.array(landmarks_2d)
                
                # Convert to pixel coordinates
                h, w = frame.shape[:2]
                landmarks_px = landmarks_2d.copy()
                landmarks_px[:, 0] *= w
                landmarks_px[:, 1] *= h
                
                # Calculate bounding box
                x_min = np.min(landmarks_px[:, 0])
                x_max = np.max(landmarks_px[:, 0])
                y_min = np.min(landmarks_px[:, 1])
                y_max = np.max(landmarks_px[:, 1])
                
                bbox = {
                    'x': x_min,
                    'y': y_min,
                    'width': x_max - x_min,
                    'height': y_max - y_min,
                    'center': [(x_min + x_max) / 2, (y_min + y_max) / 2]
                }
                
                # Get hand label (Left/Right)
                # MediaPipe gives label from camera perspective
                label = handedness.classification[0].label
                confidence = handedness.classification[0].score
                
                hand_data = {
                    'landmarks_2d': landmarks_2d,  # Normalized
                    'landmarks_px': landmarks_px,   # Pixel coordinates
                    'bbox': bbox,
                    'label': label,
                    'confidence': confidence,
                    'hand_landmarks': hand_landmarks  # Original MediaPipe object
                }
                
                hands_data.append(hand_data)
        
        # Sort hands consistently (Left hand first, then Right)
        hands_data = self._sort_hands(hands_data)
        
        return hands_data
    
    def _sort_hands(self, hands_data: List[Dict]) -> List[Dict]:
        """
        Sort hands consistently: Left first, then Right
        Uses spatial tracking to maintain identity across frames
        """
        if len(hands_data) <= 1:
            return hands_data
        
        # Sort by label: Left before Right
        hands_data.sort(key=lambda x: 0 if x['label'] == 'Left' else 1)
        
        # Additional spatial consistency check
        # If we have history, match by position continuity
        if len(self.hand_history) > 0:
            prev_hands = self.hand_history[-1]
            if len(prev_hands) == len(hands_data):
                # Calculate matching cost matrix
                cost_matrix = np.zeros((len(hands_data), len(prev_hands)))
                for i, curr_hand in enumerate(hands_data):
                    for j, prev_hand in enumerate(prev_hands):
                        # Distance between hand centers
                        dist = np.linalg.norm(
                            np.array(curr_hand['bbox']['center']) - 
                            np.array(prev_hand['bbox']['center'])
                        )
                        cost_matrix[i, j] = dist
                
                # Simple greedy matching (for 2 hands)
                if len(hands_data) == 2:
                    # Check if swap would be better
                    cost_no_swap = cost_matrix[0, 0] + cost_matrix[1, 1]
                    cost_swap = cost_matrix[0, 1] + cost_matrix[1, 0]
                    
                    if cost_swap < cost_no_swap:
                        hands_data = [hands_data[1], hands_data[0]]
        
        # Update history
        self.hand_history.append(hands_data)
        if len(self.hand_history) > self.max_history:
            self.hand_history.pop(0)
        
        return hands_data
    
    def draw_landmarks(self, frame: np.ndarray, hands_data: List[Dict]) -> np.ndarray:
        """
        Draw hand landmarks on frame
        
        Args:
            frame: BGR image
            hands_data: List of hand data from detect()
            
        Returns:
            Frame with landmarks drawn
        """
        annotated_frame = frame.copy()
        
        for hand_data in hands_data:
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_data['hand_landmarks'],
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Draw bounding box and label
            bbox = hand_data['bbox']
            color = (0, 255, 0) if hand_data['label'] == 'Left' else (0, 0, 255)
            
            cv2.rectangle(
                annotated_frame,
                (int(bbox['x']), int(bbox['y'])),
                (int(bbox['x'] + bbox['width']), int(bbox['y'] + bbox['height'])),
                color, 2
            )
            
            # Draw label
            label_text = f"{hand_data['label']} ({hand_data['confidence']:.2f})"
            cv2.putText(
                annotated_frame,
                label_text,
                (int(bbox['x']), int(bbox['y'] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        return annotated_frame
    
    def get_fingertips(self, hand_data: Dict) -> Dict[str, np.ndarray]:
        """
        Extract fingertip positions
        
        Args:
            hand_data: Hand data from detect()
            
        Returns:
            Dictionary with fingertip positions
        """
        landmarks = hand_data['landmarks_px']
        
        return {
            'thumb': landmarks[self.THUMB_TIP],
            'index': landmarks[self.INDEX_TIP],
            'middle': landmarks[self.MIDDLE_TIP],
            'ring': landmarks[self.RING_TIP],
            'pinky': landmarks[self.PINKY_TIP],
            'wrist': landmarks[self.WRIST]
        }
    
    def release(self):
        """Release MediaPipe resources"""
        self.hands.close()


