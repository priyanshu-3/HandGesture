"""
3D Hand Model with Kinematic Chain
Provides inverse kinematics and anatomical constraints
"""
import numpy as np
from typing import Dict, List, Tuple, Optional


class HandModel3D:
    """
    Kinematic hand model with joint hierarchy
    """
    
    # Joint limits (degrees) for each joint type
    JOINT_LIMITS = {
        'mcp': (-15, 90),    # Metacarpophalangeal
        'pip': (0, 110),     # Proximal interphalangeal
        'dip': (0, 90),      # Distal interphalangeal
        'thumb_cmc': (-30, 60),  # Carpometacarpal (thumb)
    }
    
    def __init__(self):
        """Initialize hand model"""
        self.joint_angles = {}
        self.landmarks_3d = None
        
    def update_from_landmarks(self, landmarks_3d: np.ndarray):
        """
        Update hand model from 3D landmarks
        
        Args:
            landmarks_3d: 21x3 array of 3D landmark positions
        """
        self.landmarks_3d = landmarks_3d.copy()
        self._compute_joint_angles()
        
    def _compute_joint_angles(self):
        """
        Compute joint angles from 3D landmarks
        Uses vectors between landmarks to estimate angles
        """
        if self.landmarks_3d is None:
            return
        
        # Finger chains (same as in depth estimator)
        chains = {
            'thumb': [0, 1, 2, 3, 4],
            'index': [0, 5, 6, 7, 8],
            'middle': [0, 9, 10, 11, 12],
            'ring': [0, 13, 14, 15, 16],
            'pinky': [0, 17, 18, 19, 20]
        }
        
        for finger_name, chain in chains.items():
            angles = []
            
            for i in range(len(chain) - 2):
                # Three consecutive points
                p1 = self.landmarks_3d[chain[i]]
                p2 = self.landmarks_3d[chain[i + 1]]
                p3 = self.landmarks_3d[chain[i + 2]]
                
                # Vectors
                v1 = p1 - p2
                v2 = p3 - p2
                
                # Angle between vectors
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                # Convert to degrees
                angle_deg = np.degrees(angle)
                angles.append(angle_deg)
            
            self.joint_angles[finger_name] = angles
    
    def get_finger_curl(self, finger_name: str) -> float:
        """
        Get curl amount for a finger (0 = extended, 1 = fully curled)
        
        Args:
            finger_name: 'thumb', 'index', 'middle', 'ring', or 'pinky'
            
        Returns:
            Curl value between 0 and 1
        """
        if finger_name not in self.joint_angles:
            return 0.0
        
        angles = self.joint_angles[finger_name]
        if len(angles) == 0:
            return 0.0
        
        # Average angle across joints
        avg_angle = np.mean(angles)
        
        # Map to 0-1 (assuming fully extended is ~180°, fully curled is ~90°)
        curl = 1.0 - (avg_angle / 180.0)
        curl = np.clip(curl, 0, 1)
        
        return curl
    
    def get_all_finger_curls(self) -> Dict[str, float]:
        """Get curl values for all fingers"""
        return {
            finger: self.get_finger_curl(finger)
            for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']
        }
    
    def is_finger_extended(self, finger_name: str, threshold: float = 0.3) -> bool:
        """
        Check if finger is extended
        
        Args:
            finger_name: Finger name
            threshold: Curl threshold (below = extended)
            
        Returns:
            True if finger is extended
        """
        curl = self.get_finger_curl(finger_name)
        return curl < threshold
    
    def is_finger_curled(self, finger_name: str, threshold: float = 0.7) -> bool:
        """
        Check if finger is curled
        
        Args:
            finger_name: Finger name
            threshold: Curl threshold (above = curled)
            
        Returns:
            True if finger is curled
        """
        curl = self.get_finger_curl(finger_name)
        return curl > threshold
    
    def get_fingertip_distance(self, finger1: str, finger2: str) -> float:
        """
        Calculate 3D distance between two fingertips
        
        Args:
            finger1, finger2: 'thumb', 'index', 'middle', 'ring', or 'pinky'
            
        Returns:
            Distance in meters
        """
        if self.landmarks_3d is None:
            return float('inf')
        
        # Fingertip indices
        tip_indices = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        idx1 = tip_indices.get(finger1)
        idx2 = tip_indices.get(finger2)
        
        if idx1 is None or idx2 is None:
            return float('inf')
        
        dist = np.linalg.norm(self.landmarks_3d[idx1] - self.landmarks_3d[idx2])
        return dist
    
    def get_fingertip_position(self, finger_name: str) -> Optional[np.ndarray]:
        """
        Get 3D position of fingertip
        
        Args:
            finger_name: Finger name
            
        Returns:
            3D position or None
        """
        if self.landmarks_3d is None:
            return None
        
        tip_indices = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        idx = tip_indices.get(finger_name)
        if idx is None:
            return None
        
        return self.landmarks_3d[idx].copy()
    
    def get_wrist_position(self) -> Optional[np.ndarray]:
        """Get 3D wrist position"""
        if self.landmarks_3d is None:
            return None
        return self.landmarks_3d[0].copy()
    
    def get_hand_orientation(self) -> Dict[str, np.ndarray]:
        """
        Compute hand orientation vectors
        
        Returns:
            Dictionary with 'palm_normal', 'palm_direction', 'side_direction'
        """
        if self.landmarks_3d is None:
            return {}
        
        # Use wrist, index MCP, and pinky MCP to define palm plane
        wrist = self.landmarks_3d[0]
        index_mcp = self.landmarks_3d[5]
        pinky_mcp = self.landmarks_3d[17]
        middle_mcp = self.landmarks_3d[9]
        
        # Palm direction (wrist to middle MCP)
        palm_dir = middle_mcp - wrist
        palm_dir = palm_dir / (np.linalg.norm(palm_dir) + 1e-6)
        
        # Side direction (pinky to index)
        side_dir = index_mcp - pinky_mcp
        side_dir = side_dir / (np.linalg.norm(side_dir) + 1e-6)
        
        # Palm normal (perpendicular to palm)
        palm_normal = np.cross(palm_dir, side_dir)
        palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-6)
        
        return {
            'palm_normal': palm_normal,
            'palm_direction': palm_dir,
            'side_direction': side_dir
        }

