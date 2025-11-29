"""
Main Application for Hand Gesture 3D Interaction
Combines all modules for real-time hand gesture recognition and 3D interaction
"""
import cv2
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import sys
from typing import Dict, Tuple

# Import custom modules
from hand_detector import HandDetector
from depth_estimator import DepthEstimator
from hand_model_3d import HandModel3D
from gesture_recognizer import GestureRecognizer
from stabilizer import LandmarkStabilizer, VelocityStabilizer
from interaction_manager import InteractionManager


class HandGesture3DApp:
    """
    Main application for hand gesture 3D interaction
    """
    
    def __init__(self, camera_id: int = 0, window_size: Tuple[int, int] = (1280, 720)):
        """
        Initialize application
        
        Args:
            camera_id: Camera device ID
            window_size: (width, height) of display window
        """
        self.window_size = window_size
        self.camera_id = camera_id
        
        # Initialize modules
        print("Initializing Hand Gesture 3D Interaction System...")
        print("=" * 60)
        
        # Hand detection
        print("✓ Loading hand detector (MediaPipe)...")
        self.hand_detector = HandDetector(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Depth estimation
        print("✓ Initializing depth estimator...")
        self.depth_estimator = DepthEstimator(focal_length=500.0)
        
        # Gesture recognition
        print("✓ Setting up gesture recognizer...")
        self.gesture_recognizer = GestureRecognizer()
        
        # Stabilizers
        print("✓ Configuring stabilization filters...")
        self.stabilizers = {
            'left': LandmarkStabilizer(smoothing=0.5),
            'right': LandmarkStabilizer(smoothing=0.5)
        }
        
        # 3D hand models
        self.hand_models = {
            'left': HandModel3D(),
            'right': HandModel3D()
        }
        
        # Interaction manager
        print("✓ Initializing 3D interaction manager...")
        self.interaction_manager = InteractionManager(window_size)
        
        # Camera
        print(f"✓ Opening camera (device {camera_id})...")
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # Performance metrics
        self.fps = 0
        self.frame_times = []
        
        # UI state
        self.show_debug = True
        self.show_3d_view = True
        self.paused = False
        
        # Initialize Pygame and OpenGL
        self._init_opengl()
        
        print("=" * 60)
        print("✓ Initialization complete!")
        print("\nControls:")
        print("  SPACE - Pause/Resume")
        print("  D     - Toggle debug view")
        print("  3     - Toggle 3D view")
        print("  R     - Reset camera")
        print("  Q/ESC - Quit")
        print("\nGestures:")
        print("  PINCH    - Grab and move objects")
        print("  OPEN     - Release objects")
        print("  FIST     - Rotate objects")
        print("  POINT    - Point at objects")
        print("  TWO-HAND - Scale objects")
        print("=" * 60)
    
    def _init_opengl(self):
        """Initialize Pygame and OpenGL for 3D rendering"""
        pygame.init()
        
        # Create OpenGL window
        pygame.display.set_mode(self.window_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Hand Gesture 3D Interaction")
        
        # OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Light position
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.7, 0.7, 0.7, 1])
        
        # Projection
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.window_size[0] / self.window_size[1]), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Background color
        glClearColor(0.1, 0.1, 0.15, 1.0)
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process single frame
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Dictionary with processing results
        """
        # Detect hands
        hands_data = self.hand_detector.detect(frame)
        
        # Process each hand
        processed_hands = {}
        
        for hand_data in hands_data:
            label = hand_data['label'].lower()
            
            # Estimate depth and get 3D landmarks
            depth_result = self.depth_estimator.estimate_depth(
                hand_data, 
                frame.shape[:2]
            )
            
            landmarks_3d = depth_result['landmarks_3d']
            
            # Apply stabilization
            if label in self.stabilizers:
                landmarks_3d = self.stabilizers[label].update(landmarks_3d)
            
            # Update 3D hand model
            self.hand_models[label].update_from_landmarks(landmarks_3d)
            
            # Recognize gesture
            gesture_info = self.gesture_recognizer.recognize(
                self.hand_models[label],
                hand_data['label']
            )
            
            processed_hands[label] = {
                'hand_data': hand_data,
                'landmarks_3d': landmarks_3d,
                'depth_info': depth_result,
                'gesture_info': gesture_info,
                'hand_model': self.hand_models[label]
            }
        
        # Check for two-hand gestures
        two_hand_gesture = None
        if 'left' in processed_hands and 'right' in processed_hands:
            two_hand_gesture = self.gesture_recognizer.detect_two_hand_gestures(
                processed_hands['left']['gesture_info'],
                processed_hands['right']['gesture_info']
            )
            
            if two_hand_gesture is not None:
                # Handle two-hand interaction
                self.interaction_manager.handle_two_hand_interaction(
                    two_hand_gesture['left_pos'],
                    two_hand_gesture['right_pos'],
                    two_hand_gesture['gesture']
                )
        
        # Update interactions
        self.interaction_manager.update(processed_hands)
        
        return {
            'hands': processed_hands,
            'two_hand_gesture': two_hand_gesture,
            'num_hands': len(hands_data)
        }
    
    def render_debug_view(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Render debug information on frame
        
        Args:
            frame: Original frame
            results: Processing results
            
        Returns:
            Annotated frame
        """
        debug_frame = frame.copy()
        
        # Draw landmarks for each hand
        for label, hand_info in results['hands'].items():
            hand_data = hand_info['hand_data']
            depth_info = hand_info['depth_info']
            gesture_info = hand_info['gesture_info']
            
            # Draw 2D landmarks
            debug_frame = self.hand_detector.draw_landmarks(
                debug_frame,
                [hand_data]
            )
            
            # Draw depth-colored fingertips
            landmarks_px = hand_data['landmarks_px']
            landmarks_3d = hand_info['landmarks_3d']
            
            fingertip_indices = [4, 8, 12, 16, 20]
            for idx in fingertip_indices:
                x, y = int(landmarks_px[idx, 0]), int(landmarks_px[idx, 1])
                depth = landmarks_3d[idx, 2]
                
                color = self.depth_estimator.depth_to_color(depth)
                cv2.circle(debug_frame, (x, y), 8, color, -1)
                cv2.circle(debug_frame, (x, y), 8, (255, 255, 255), 2)
            
            # Draw gesture info
            bbox = hand_data['bbox']
            gesture = gesture_info['gesture']
            confidence = gesture_info['confidence']
            
            gesture_text = self.gesture_recognizer.get_gesture_description(gesture)
            text_y = int(bbox['y'] + bbox['height'] + 30)
            
            cv2.putText(
                debug_frame,
                f"{gesture_text} ({confidence:.2f})",
                (int(bbox['x']), text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            
            # Draw depth info
            depth_text = f"Depth: {depth_info['base_depth']:.2f}m"
            cv2.putText(
                debug_frame,
                depth_text,
                (int(bbox['x']), text_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )
        
        # Draw FPS and info
        info_y = 30
        cv2.putText(debug_frame, f"FPS: {self.fps:.1f}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        info_y += 30
        cv2.putText(debug_frame, f"Hands: {results['num_hands']}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Two-hand gesture
        if results['two_hand_gesture'] is not None:
            info_y += 30
            gesture_name = results['two_hand_gesture']['gesture']
            cv2.putText(debug_frame, f"Two-Hand: {gesture_name}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        return debug_frame
    
    def run(self):
        """Main application loop"""
        clock = pygame.time.Clock()
        
        try:
            while True:
                frame_start = time.time()
                
                # Handle Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                            return
                        elif event.key == pygame.K_SPACE:
                            self.paused = not self.paused
                        elif event.key == pygame.K_d:
                            self.show_debug = not self.show_debug
                        elif event.key == pygame.K_3:
                            self.show_3d_view = not self.show_3d_view
                        elif event.key == pygame.K_r:
                            self.interaction_manager.camera_rotation = [20, 45]
                
                if not self.paused:
                    # Capture frame
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to capture frame")
                        break
                    
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame
                    results = self.process_frame(frame)
                    
                    # Render debug view
                    if self.show_debug:
                        debug_frame = self.render_debug_view(frame, results)
                        cv2.imshow('Hand Gesture Debug', debug_frame)
                
                # Render 3D view
                if self.show_3d_view:
                    self.interaction_manager.draw_scene()
                    pygame.display.flip()
                
                # Handle OpenCV window
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)
                self.fps = 1.0 / (np.mean(self.frame_times) + 1e-6)
                
                # Limit frame rate
                clock.tick(60)
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        self.hand_detector.release()
        print("Done!")


def main():
    """Entry point"""
    print("\n" + "=" * 60)
    print(" Hand Gesture 3D Interaction System")
    print(" Monocular depth estimation + 3D interaction")
    print("=" * 60 + "\n")
    
    try:
        app = HandGesture3DApp(camera_id=0, window_size=(1280, 720))
        app.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()

