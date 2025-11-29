"""
Demo script to test individual modules
Useful for understanding and debugging the system
"""
import cv2
import numpy as np
import time

from hand_detector import HandDetector
from depth_estimator import DepthEstimator
from hand_model_3d import HandModel3D
from gesture_recognizer import GestureRecognizer
from stabilizer import LandmarkStabilizer


def demo_hand_detection():
    """Demo 1: Hand detection and 2D landmarks"""
    print("\n" + "="*60)
    print("DEMO 1: Hand Detection & 2D Landmarks")
    print("="*60)
    print("Shows: Hand detection, bounding boxes, left/right labeling")
    print("Press 'q' to exit this demo")
    print("="*60 + "\n")
    
    detector = HandDetector(max_num_hands=2)
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            hands_data = detector.detect(frame)
            
            # Draw landmarks
            annotated = detector.draw_landmarks(frame, hands_data)
            
            # Show info
            for hand in hands_data:
                print(f"{hand['label']} hand detected - "
                      f"Confidence: {hand['confidence']:.2f} - "
                      f"Bbox: {hand['bbox']['width']:.0f}x{hand['bbox']['height']:.0f}px")
            
            cv2.imshow('Demo 1: Hand Detection', annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()


def demo_depth_estimation():
    """Demo 2: Depth estimation visualization"""
    print("\n" + "="*60)
    print("DEMO 2: Depth Estimation")
    print("="*60)
    print("Shows: 3D depth color-coded on fingertips")
    print("Blue = far, Green = medium, Red = close")
    print("Press 'q' to exit this demo")
    print("="*60 + "\n")
    
    detector = HandDetector(max_num_hands=2)
    depth_estimator = DepthEstimator()
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            hands_data = detector.detect(frame)
            
            annotated = frame.copy()
            
            for hand_data in hands_data:
                # Estimate depth
                depth_result = depth_estimator.estimate_depth(
                    hand_data,
                    frame.shape[:2]
                )
                
                landmarks_px = hand_data['landmarks_px']
                landmarks_3d = depth_result['landmarks_3d']
                
                # Draw depth-colored fingertips
                fingertip_indices = [4, 8, 12, 16, 20]
                for idx in fingertip_indices:
                    x, y = int(landmarks_px[idx, 0]), int(landmarks_px[idx, 1])
                    depth = landmarks_3d[idx, 2]
                    
                    color = depth_estimator.depth_to_color(depth)
                    cv2.circle(annotated, (x, y), 12, color, -1)
                    cv2.circle(annotated, (x, y), 12, (255, 255, 255), 2)
                    
                    # Show depth value
                    cv2.putText(annotated, f"{depth:.2f}m", 
                              (x-30, y-20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Show base depth
                bbox = hand_data['bbox']
                cv2.putText(annotated, 
                          f"{hand_data['label']}: {depth_result['base_depth']:.2f}m",
                          (int(bbox['x']), int(bbox['y'] - 10)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Demo 2: Depth Estimation', annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()


def demo_gesture_recognition():
    """Demo 3: Gesture recognition"""
    print("\n" + "="*60)
    print("DEMO 3: Gesture Recognition")
    print("="*60)
    print("Try these gestures:")
    print("  - PINCH (thumb + index)")
    print("  - FIST (close all fingers)")
    print("  - OPEN PALM (extend all fingers)")
    print("  - POINT (index only)")
    print("  - PEACE (index + middle)")
    print("  - OK (thumb + middle)")
    print("Press 'q' to exit this demo")
    print("="*60 + "\n")
    
    detector = HandDetector(max_num_hands=2)
    depth_estimator = DepthEstimator()
    gesture_recognizer = GestureRecognizer()
    
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            hands_data = detector.detect(frame)
            
            annotated = detector.draw_landmarks(frame, hands_data)
            
            for hand_data in hands_data:
                # Get 3D landmarks
                depth_result = depth_estimator.estimate_depth(
                    hand_data,
                    frame.shape[:2]
                )
                
                # Create hand model
                hand_model = HandModel3D()
                hand_model.update_from_landmarks(depth_result['landmarks_3d'])
                
                # Recognize gesture
                gesture_info = gesture_recognizer.recognize(
                    hand_model,
                    hand_data['label']
                )
                
                # Display gesture
                bbox = hand_data['bbox']
                gesture = gesture_info['gesture']
                confidence = gesture_info['confidence']
                
                gesture_desc = gesture_recognizer.get_gesture_description(gesture)
                
                # Big text for gesture
                text_y = int(bbox['y'] + bbox['height'] + 40)
                cv2.putText(annotated, 
                          f"{gesture_desc}",
                          (int(bbox['x']), text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                
                cv2.putText(annotated, 
                          f"Confidence: {confidence:.2f}",
                          (int(bbox['x']), text_y + 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                print(f"{hand_data['label']}: {gesture_desc} ({confidence:.2f})")
            
            cv2.imshow('Demo 3: Gesture Recognition', annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()


def demo_stabilization():
    """Demo 4: Stabilization comparison"""
    print("\n" + "="*60)
    print("DEMO 4: Stabilization")
    print("="*60)
    print("Shows: Raw vs. stabilized landmarks")
    print("Left window: Raw, Right window: Stabilized")
    print("Press 'q' to exit this demo")
    print("="*60 + "\n")
    
    detector = HandDetector(max_num_hands=1)
    stabilizer = LandmarkStabilizer(smoothing=0.7)
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            hands_data = detector.detect(frame)
            
            if len(hands_data) > 0:
                hand_data = hands_data[0]
                landmarks_px = hand_data['landmarks_px']
                
                # Apply stabilization
                landmarks_stabilized = stabilizer.update(landmarks_px)
                
                # Draw both
                raw_frame = frame.copy()
                stable_frame = frame.copy()
                
                # Draw raw
                for i, (x, y, _) in enumerate(landmarks_px):
                    cv2.circle(raw_frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                
                # Draw stabilized
                for i, (x, y, _) in enumerate(landmarks_stabilized):
                    cv2.circle(stable_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                
                cv2.putText(raw_frame, "RAW (jittery)", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(stable_frame, "STABILIZED (smooth)", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show side by side
                combined = np.hstack([raw_frame, stable_frame])
                cv2.imshow('Demo 4: Stabilization', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("  Hand Gesture 3D Interaction - Module Demos")
    print("="*60)
    print("\nSelect a demo to run:")
    print("  1 - Hand Detection & 2D Landmarks")
    print("  2 - Depth Estimation (3D)")
    print("  3 - Gesture Recognition")
    print("  4 - Stabilization Comparison")
    print("  5 - Run all demos sequentially")
    print("  0 - Exit")
    print("="*60)
    
    choice = input("\nEnter choice (0-5): ").strip()
    
    if choice == '1':
        demo_hand_detection()
    elif choice == '2':
        demo_depth_estimation()
    elif choice == '3':
        demo_gesture_recognition()
    elif choice == '4':
        demo_stabilization()
    elif choice == '5':
        print("\nRunning all demos...")
        demo_hand_detection()
        demo_depth_estimation()
        demo_gesture_recognition()
        demo_stabilization()
    elif choice == '0':
        print("Exiting...")
        return
    else:
        print("Invalid choice!")
        return
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("To run the full application: python main.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


