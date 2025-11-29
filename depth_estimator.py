"""
Monocular Depth Estimation Module
Estimates relative 3D depth from 2D landmarks using geometric priors
"""
import numpy as np
from scipy.optimize import least_squares
from typing import Dict, List, Optional, Tuple


class DepthEstimator:
    """
    Estimates depth for hand landmarks using geometric constraints
    and anthropometric priors
    """
    
    # Average hand bone lengths (normalized, thumb to pinky)
    # Based on anthropometric data
    BONE_LENGTHS = {
        'palm_width': 1.0,  # Reference length
        'thumb': [0.38, 0.32, 0.27],  # CMC, MCP, IP
        'index': [0.43, 0.27, 0.18],  # MCP, PIP, DIP
        'middle': [0.48, 0.30, 0.20],
        'ring': [0.45, 0.27, 0.18],
        'pinky': [0.38, 0.20, 0.15]
    }
    
    # Landmark indices for bone segments
    FINGER_CHAINS = {
        'thumb': [1, 2, 3, 4],
        'index': [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring': [13, 14, 15, 16],
        'pinky': [17, 18, 19, 20]
    }
    
    def __init__(self, focal_length: float = 500.0, baseline_depth: float = 0.5):
        """
        Initialize depth estimator
        
        Args:
            focal_length: Approximate camera focal length in pixels
            baseline_depth: Initial depth estimate in meters
        """
        self.focal_length = focal_length
        self.baseline_depth = baseline_depth
        
        # Camera intrinsics (simplified pinhole model)
        self.cx = 320  # Principal point x (will be updated)
        self.cy = 240  # Principal point y (will be updated)
        
    def set_frame_size(self, width: int, height: int):
        """Update camera parameters based on frame size"""
        self.cx = width / 2
        self.cy = height / 2
        
    def estimate_depth(self, hand_data: Dict, frame_shape: Tuple[int, int]) -> Dict:
        """
        Estimate 3D positions for hand landmarks
        
        Args:
            hand_data: Hand data from detector (with 2D landmarks)
            frame_shape: (height, width) of the frame
            
        Returns:
            Dictionary with 3D landmarks and depth information
        """
        landmarks_2d = hand_data['landmarks_px']
        bbox = hand_data['bbox']
        
        # Update camera center
        h, w = frame_shape[:2]
        self.set_frame_size(w, h)
        
        # Step 1: Estimate global scale from bounding box size
        scale = self._estimate_scale_from_bbox(bbox, w)
        
        # Step 2: Estimate depth using perspective and scale
        base_depth = self._estimate_base_depth(bbox, scale)
        
        # Step 3: Compute per-landmark depth using geometric constraints
        landmarks_3d = self._compute_3d_landmarks(
            landmarks_2d, 
            base_depth, 
            scale
        )
        
        # Step 4: Refine using kinematic constraints
        landmarks_3d = self._refine_with_constraints(landmarks_2d, landmarks_3d, scale)
        
        result = {
            'landmarks_3d': landmarks_3d,
            'base_depth': base_depth,
            'scale': scale,
            'landmarks_2d': landmarks_2d
        }
        
        return result
    
    def _estimate_scale_from_bbox(self, bbox: Dict, frame_width: int) -> float:
        """
        Estimate hand scale from bounding box size
        Larger bbox -> closer to camera
        """
        # Typical hand width is ~80-100mm
        # Use bbox width as proxy for hand size
        bbox_width = bbox['width']
        
        # Normalize by frame width (larger frame = larger bbox for same distance)
        normalized_width = bbox_width / frame_width
        
        # Map to scale factor (empirical relationship)
        # At 0.5m: hand ~200px in 640px frame -> normalized_width ~ 0.3
        scale = normalized_width / 0.25
        scale = np.clip(scale, 0.5, 2.5)
        
        return scale
    
    def _estimate_base_depth(self, bbox: Dict, scale: float) -> float:
        """
        Estimate base depth (wrist depth) using perspective projection
        
        Z = (f * real_width) / pixel_width
        """
        # Average hand width: ~85mm = 0.085m
        real_hand_width = 0.085
        
        bbox_width_px = bbox['width']
        
        # Perspective projection formula
        depth = (self.focal_length * real_hand_width * scale) / (bbox_width_px + 1e-6)
        
        # Clamp to reasonable range (0.2m to 2.0m)
        depth = np.clip(depth, 0.2, 2.0)
        
        return depth
    
    def _compute_3d_landmarks(self, 
                              landmarks_2d: np.ndarray, 
                              base_depth: float,
                              scale: float) -> np.ndarray:
        """
        Convert 2D landmarks to 3D using estimated depth
        
        Args:
            landmarks_2d: 2D landmarks in pixel coordinates (N x 3)
            base_depth: Base depth estimate for wrist
            scale: Global scale factor
            
        Returns:
            3D landmarks (N x 3) in camera coordinates (meters)
        """
        landmarks_3d = np.zeros((21, 3))
        
        # Wrist (landmark 0) is at base depth
        wrist_2d = landmarks_2d[0, :2]
        
        for i in range(21):
            x_px, y_px = landmarks_2d[i, :2]
            
            # Estimate depth variation based on distance from wrist
            # Points further from wrist in 2D are assumed to protrude forward
            dist_from_wrist_2d = np.linalg.norm(landmarks_2d[i, :2] - wrist_2d)
            
            # MediaPipe already provides a rough depth estimate in landmarks_2d[:, 2]
            # Use it as a relative depth cue
            relative_z = landmarks_2d[i, 2] * scale * 0.1  # Scale factor
            
            # Compute depth for this landmark
            depth = base_depth + relative_z
            depth = max(depth, 0.1)  # Minimum depth
            
            # Back-project to 3D using pinhole camera model
            x_3d = (x_px - self.cx) * depth / self.focal_length
            y_3d = (y_px - self.cy) * depth / self.focal_length
            z_3d = depth
            
            landmarks_3d[i] = [x_3d, y_3d, z_3d]
        
        return landmarks_3d
    
    def _refine_with_constraints(self, 
                                 landmarks_2d: np.ndarray,
                                 landmarks_3d: np.ndarray,
                                 scale: float) -> np.ndarray:
        """
        Refine 3D landmarks using bone length constraints
        
        Uses nonlinear optimization to enforce anatomical constraints
        """
        # For real-time performance, use simplified constraint enforcement
        refined = landmarks_3d.copy()
        
        # Enforce bone length constraints for each finger
        for finger_name, chain in self.FINGER_CHAINS.items():
            bone_lengths = self.BONE_LENGTHS.get(finger_name, [0.3, 0.25, 0.2])
            
            for i in range(len(chain) - 1):
                idx1, idx2 = chain[i], chain[i + 1]
                
                # Current 3D distance
                current_dist = np.linalg.norm(refined[idx2] - refined[idx1])
                
                # Target distance (scaled by global scale)
                target_dist = bone_lengths[i] * scale * 0.08  # Convert to meters
                
                # Adjust second point to match target distance
                if current_dist > 1e-6:
                    direction = (refined[idx2] - refined[idx1]) / current_dist
                    refined[idx2] = refined[idx1] + direction * target_dist
        
        return refined
    
    def depth_to_color(self, depth: float, min_depth: float = 0.2, 
                      max_depth: float = 1.5) -> tuple:
        """
        Convert depth value to color for visualization
        
        Returns:
            (B, G, R) color tuple
        """
        # Normalize depth to 0-1
        normalized = (depth - min_depth) / (max_depth - min_depth)
        normalized = np.clip(normalized, 0, 1)
        
        # Map to color: blue (far) -> green -> red (near)
        if normalized < 0.5:
            # Blue to green
            t = normalized * 2
            b = int(255 * (1 - t))
            g = int(255 * t)
            r = 0
        else:
            # Green to red
            t = (normalized - 0.5) * 2
            b = 0
            g = int(255 * (1 - t))
            r = int(255 * t)
        
        return (b, g, r)

