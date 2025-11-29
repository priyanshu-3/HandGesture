# Quick Start Guide

Get up and running in 2 minutes!

## Step 1: Install Dependencies

### Option A: Automatic (macOS/Linux)
```bash
chmod +x run.sh
./run.sh
```

### Option B: Manual
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

## Step 2: Run the Application

```bash
python main.py
```

## Step 3: Try Some Gestures!

### Basic Gestures
1. **Show your hand** to the camera (palm facing camera)
2. **Pinch** (thumb + index together) to grab the colored cubes
3. **Open palm** to release
4. **Make a fist** to rotate
5. **Point** with index finger

### Two-Hand Gestures
1. Show **both hands**
2. Make **pinch** with both hands
3. **Open both palms** and move apart/together to scale

## Troubleshooting

### Camera not working?
- Check camera permissions
- Try a different camera ID in `main.py`:
  ```python
  app = HandGesture3DApp(camera_id=1)  # Try 0, 1, 2...
  ```

### Low FPS?
- Press `D` to disable debug view
- Close other applications
- Reduce lighting complexity

### Hands not detected?
- Ensure good lighting
- Keep hands in frame
- Distance: 0.3m - 1.5m from camera works best

## Next Steps

- Read `README.md` for full documentation
- Check `ROADMAP.md` to understand the architecture
- Customize gestures in `gesture_recognizer.py`
- Add your own 3D objects in `interaction_manager.py`

---

**Enjoy! 🖐️✨**


