"""
3D Interaction Manager
Handles interaction logic and 3D scene manipulation
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


class Object3D:
    """Represents a 3D object in the scene"""
    
    def __init__(self, position: np.ndarray, size: float = 0.1, 
                 color: Tuple[float, float, float] = (1.0, 0.5, 0.0)):
        """
        Initialize 3D object
        
        Args:
            position: Initial position (x, y, z)
            size: Object size
            color: RGB color (0-1 range)
        """
        self.position = position.copy()
        self.size = size
        self.color = color
        self.rotation = np.array([0.0, 0.0, 0.0])  # Euler angles
        self.selected = False
        
    def draw(self):
        """Draw object using OpenGL"""
        glPushMatrix()
        
        # Apply transformations
        glTranslatef(*self.position)
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)
        
        # Set color (brighter if selected)
        if self.selected:
            glColor3f(self.color[0] * 1.5, self.color[1] * 1.5, self.color[2] * 1.5)
        else:
            glColor3f(*self.color)
        
        # Draw cube
        s = self.size
        self._draw_cube(s)
        
        glPopMatrix()
    
    def _draw_cube(self, size: float):
        """Draw a cube"""
        s = size / 2
        
        glBegin(GL_QUADS)
        
        # Front face
        glVertex3f(-s, -s, s)
        glVertex3f(s, -s, s)
        glVertex3f(s, s, s)
        glVertex3f(-s, s, s)
        
        # Back face
        glVertex3f(-s, -s, -s)
        glVertex3f(-s, s, -s)
        glVertex3f(s, s, -s)
        glVertex3f(s, -s, -s)
        
        # Top face
        glVertex3f(-s, s, -s)
        glVertex3f(-s, s, s)
        glVertex3f(s, s, s)
        glVertex3f(s, s, -s)
        
        # Bottom face
        glVertex3f(-s, -s, -s)
        glVertex3f(s, -s, -s)
        glVertex3f(s, -s, s)
        glVertex3f(-s, -s, s)
        
        # Right face
        glVertex3f(s, -s, -s)
        glVertex3f(s, s, -s)
        glVertex3f(s, s, s)
        glVertex3f(s, -s, s)
        
        # Left face
        glVertex3f(-s, -s, -s)
        glVertex3f(-s, -s, s)
        glVertex3f(-s, s, s)
        glVertex3f(-s, s, -s)
        
        glEnd()
    
    def is_point_inside(self, point: np.ndarray, tolerance: float = 1.5) -> bool:
        """Check if point is inside object (with tolerance)"""
        dist = np.linalg.norm(point - self.position)
        return dist < (self.size * tolerance)


class InteractionManager:
    """
    Manages 3D interactions with hand gestures
    """
    
    def __init__(self, window_size: Tuple[int, int] = (800, 600)):
        """
        Initialize interaction manager
        
        Args:
            window_size: (width, height) of the window
        """
        self.window_size = window_size
        
        # 3D objects in scene
        self.objects = []
        self._create_initial_objects()
        
        # Interaction state
        self.selected_object = None
        self.interaction_mode = None  # 'translate', 'rotate', 'scale'
        self.grab_offset = np.zeros(3)
        
        # Virtual hand representations
        self.virtual_hands = {}
        
        # Camera parameters
        self.camera_distance = 2.0
        self.camera_rotation = [20, 45]  # [pitch, yaw]
        
    def _create_initial_objects(self):
        """Create initial 3D objects in scene"""
        # Create a few cubes
        self.objects.append(Object3D(np.array([0.0, 0.0, 0.5]), 0.1, (1.0, 0.3, 0.3)))
        self.objects.append(Object3D(np.array([0.2, 0.1, 0.6]), 0.08, (0.3, 1.0, 0.3)))
        self.objects.append(Object3D(np.array([-0.2, -0.1, 0.55]), 0.12, (0.3, 0.3, 1.0)))
    
    def update(self, hands_data: Dict):
        """
        Update interactions based on hand data
        
        Args:
            hands_data: Dictionary with hand information
                {
                    'left': {'gesture': ..., 'landmarks_3d': ..., ...},
                    'right': {'gesture': ..., 'landmarks_3d': ..., ...}
                }
        """
        # Update virtual hands
        self.virtual_hands = {}
        
        for hand_label in ['left', 'right']:
            if hand_label in hands_data:
                hand_info = hands_data[hand_label]
                
                # Get hand position (wrist or fingertip depending on gesture)
                landmarks_3d = hand_info.get('landmarks_3d')
                gesture_info = hand_info.get('gesture_info', {})
                gesture = gesture_info.get('gesture', 'none')
                
                if landmarks_3d is not None:
                    # Use index fingertip for pointing/selection
                    hand_pos = landmarks_3d[8]  # Index fingertip
                    
                    self.virtual_hands[hand_label] = {
                        'position': hand_pos,
                        'gesture': gesture,
                        'landmarks_3d': landmarks_3d
                    }
                    
                    # Handle interactions based on gesture
                    self._handle_gesture_interaction(hand_label, hand_pos, gesture_info)
    
    def _handle_gesture_interaction(self, hand_label: str, 
                                    hand_pos: np.ndarray, 
                                    gesture_info: Dict):
        """
        Handle interaction based on gesture
        
        Args:
            hand_label: 'left' or 'right'
            hand_pos: 3D position of hand
            gesture_info: Gesture information
        """
        gesture = gesture_info.get('gesture', 'none')
        
        if gesture == 'pinch':
            # Pinch to grab and move object
            if self.selected_object is None:
                # Try to select object
                for obj in self.objects:
                    if obj.is_point_inside(hand_pos):
                        self.selected_object = obj
                        obj.selected = True
                        self.grab_offset = obj.position - hand_pos
                        self.interaction_mode = 'translate'
                        break
            else:
                # Move selected object
                self.selected_object.position = hand_pos + self.grab_offset
        
        elif gesture == 'open_palm':
            # Release object
            if self.selected_object is not None:
                self.selected_object.selected = False
                self.selected_object = None
                self.interaction_mode = None
        
        elif gesture == 'fist':
            # Fist to rotate object
            if self.selected_object is not None:
                # Rotate based on hand movement
                self.interaction_mode = 'rotate'
                # Simple rotation (can be enhanced)
                self.selected_object.rotation[1] += 2.0
    
    def handle_two_hand_interaction(self, left_hand_pos: np.ndarray, 
                                   right_hand_pos: np.ndarray,
                                   gesture_type: str):
        """
        Handle two-hand gestures
        
        Args:
            left_hand_pos: Left hand position
            right_hand_pos: Right hand position
            gesture_type: Type of two-hand gesture
        """
        if gesture_type == 'two_hand_scale' and self.selected_object is not None:
            # Scale object based on hand distance
            distance = np.linalg.norm(right_hand_pos - left_hand_pos)
            self.selected_object.size = distance * 0.5
            self.selected_object.size = np.clip(self.selected_object.size, 0.05, 0.3)
    
    def draw_scene(self):
        """Draw the 3D scene"""
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up camera
        glLoadIdentity()
        gluLookAt(
            self.camera_distance * np.sin(np.radians(self.camera_rotation[1])),
            self.camera_distance * np.sin(np.radians(self.camera_rotation[0])),
            self.camera_distance * np.cos(np.radians(self.camera_rotation[1])),
            0, 0, 0.5,
            0, 1, 0
        )
        
        # Draw grid
        self._draw_grid()
        
        # Draw objects
        for obj in self.objects:
            obj.draw()
        
        # Draw virtual hands
        self._draw_virtual_hands()
    
    def _draw_grid(self):
        """Draw reference grid"""
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_LINES)
        
        size = 1.0
        step = 0.1
        
        for i in np.arange(-size, size + step, step):
            # Lines parallel to X
            glVertex3f(-size, 0, i)
            glVertex3f(size, 0, i)
            
            # Lines parallel to Z
            glVertex3f(i, 0, -size)
            glVertex3f(i, 0, size)
        
        glEnd()
        
        # Draw axes
        glBegin(GL_LINES)
        
        # X axis (red)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0.3, 0, 0)
        
        # Y axis (green)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0.3, 0)
        
        # Z axis (blue)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 0.3)
        
        glEnd()
    
    def _draw_virtual_hands(self):
        """Draw virtual hand representations"""
        for hand_label, hand_data in self.virtual_hands.items():
            landmarks_3d = hand_data['landmarks_3d']
            
            # Draw hand skeleton
            color = (0.0, 1.0, 0.0) if hand_label == 'left' else (1.0, 0.0, 0.0)
            glColor3f(*color)
            
            # Draw fingertips as spheres
            fingertip_indices = [4, 8, 12, 16, 20]
            for idx in fingertip_indices:
                pos = landmarks_3d[idx]
                self._draw_sphere(pos, 0.01)
            
            # Draw connections
            connections = [
                [0, 1, 2, 3, 4],  # Thumb
                [0, 5, 6, 7, 8],  # Index
                [0, 9, 10, 11, 12],  # Middle
                [0, 13, 14, 15, 16],  # Ring
                [0, 17, 18, 19, 20]  # Pinky
            ]
            
            glBegin(GL_LINES)
            for chain in connections:
                for i in range(len(chain) - 1):
                    glVertex3fv(landmarks_3d[chain[i]])
                    glVertex3fv(landmarks_3d[chain[i + 1]])
            glEnd()
    
    def _draw_sphere(self, position: np.ndarray, radius: float):
        """Draw a sphere at position"""
        glPushMatrix()
        glTranslatef(*position)
        
        quad = gluNewQuadric()
        gluSphere(quad, radius, 10, 10)
        gluDeleteQuadric(quad)
        
        glPopMatrix()
    
    def add_object(self, position: np.ndarray, size: float = 0.1):
        """Add new object to scene"""
        color = np.random.rand(3) * 0.5 + 0.5  # Random pastel color
        obj = Object3D(position, size, tuple(color))
        self.objects.append(obj)


