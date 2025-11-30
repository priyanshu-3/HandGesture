# Troubleshooting Guide

## 🔍 Step-by-Step Diagnosis

### Step 1: Check Camera Access

**Run the camera test:**
```bash
cd /Users/priyanshumehra/Desktop/HandGesture
source venv/bin/activate
python test_camera.py
```

**Expected Result:** A window should appear showing your camera feed

**If it fails:**
- Check System Settings → Privacy & Security → Camera
- Enable for your terminal app
- Close other apps using the camera (Zoom, FaceTime, etc.)

---

### Step 2: Test Hand Detection

**Run the simple test:**
```bash
python simple_test.py
```

**Expected Result:** 
- Window appears with camera feed
- When you show your hand, green lines appear (landmarks)
- Text says "Hand Detected!"

**If hand not detected:**
- Improve lighting (face a window or turn on lights)
- Get closer (30-60cm from camera)
- Show full palm to camera
- Spread fingers apart

---

### Step 3: Test Gesture Recognition

**In the simple test window:**
1. Show your hand (palm facing camera)
2. Touch thumb and index finger together (PINCH)
3. You should see "PINCH DETECTED!" in green

**If pinch not detected:**
- Make sure thumb and index are actually touching
- Move hand closer to camera
- Keep other fingers visible

---

### Step 4: Run Full Application

**Run main app:**
```bash
python main.py
```

**Expected Result:**
- TWO windows appear:
  1. "Hand Gesture Debug" - shows camera with landmarks
  2. "Hand Gesture 3D Interaction" - shows 3D scene with cubes

**If only one window appears:**
- Check other desktop spaces (swipe left/right with 3 fingers)
- Check Mission Control (swipe up with 3 fingers)
- Look for minimized windows in Dock

---

## 🎯 Common Issues & Solutions

### Issue: "Failed to capture frame"

**Cause:** Camera in use or no permission

**Fix:**
1. Close all apps using camera (Zoom, Teams, FaceTime)
2. Check System Settings → Privacy & Security → Camera
3. Restart Terminal

---

### Issue: Window appears but black screen

**Cause:** Camera permission denied

**Fix:**
1. System Settings → Privacy & Security → Camera
2. Enable for Terminal/iTerm
3. Restart application

---

### Issue: Hand detected but can't grab cubes

**Cause:** Coordinate mismatch (FIXED in latest version)

**Fix:** Make sure you're running the updated code:
```bash
git pull  # If using git
python main.py  # Restart app
```

**To verify fix worked:**
- You should see hand skeleton in 3D window
- Hand skeleton should overlap with cubes
- Pinch gesture near cube should highlight it

---

### Issue: Cubes too far away

**Cause:** Depth calibration

**Temporary Fix:** Adjust distance from camera
- Try 40-50cm (arm's length)
- Move slightly forward/backward until hand aligns with cubes

**Code Fix:** Edit `interaction_manager.py` line 137-139:
```python
# Change cube Z positions (current: 0.5)
self.objects.append(Object3D(np.array([0.0, 0.0, 0.4]), ...))  # Closer
# or
self.objects.append(Object3D(np.array([0.0, 0.0, 0.6]), ...))  # Further
```

---

### Issue: Application slow/laggy

**Cause:** High CPU usage from MediaPipe

**Fix:**
1. Close other applications
2. Lower detection confidence in `hand_detector.py`:
   ```python
   min_detection_confidence=0.5  # Lower = faster but less accurate
   ```
3. Reduce frame resolution in `main.py`:
   ```python
   self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
   self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
   ```

---

### Issue: Jittery hand movements

**Cause:** Insufficient smoothing

**Fix:** Increase smoothing in `main.py` line ~80:
```python
self.stabilizers = {
    'left': LandmarkStabilizer(smoothing=0.8),  # Increase from 0.5
    'right': LandmarkStabilizer(smoothing=0.8)
}
```

---

## 📊 Performance Expectations

### Normal Performance:
- **FPS**: 30-60 fps
- **CPU**: 40-80% (one core)
- **RAM**: ~300-500 MB
- **Latency**: 30-50ms

### If performance is worse:
- Close other applications
- Check Activity Monitor for background processes
- Ensure good lighting (bad lighting = more processing)

---

## 🆘 Still Not Working?

### Debug Mode:

**Run with verbose output:**
```bash
python main.py 2>&1 | tee debug.log
```

This saves all output to `debug.log` for analysis.

### Check what's running:
```bash
ps aux | grep python
```

### Kill hanging processes:
```bash
pkill -9 Python
```

### Reinstall dependencies:
```bash
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ✅ Verification Checklist

Before asking for help, verify:
- [ ] Camera works (test_camera.py shows video)
- [ ] Hands detected (simple_test.py shows green landmarks)
- [ ] Pinch recognized (simple_test.py shows "PINCH DETECTED")
- [ ] Two windows appear (Debug + 3D)
- [ ] Hand skeleton visible in 3D window
- [ ] FPS > 20 in debug window
- [ ] Good lighting (face toward light source)
- [ ] Correct distance (30-80cm from camera)

---

## 📞 Getting Help

If still not working, provide:
1. Output of `python simple_test.py`
2. Screenshot of both windows (if they appear)
3. macOS version (`sw_vers`)
4. Python version (`python --version`)
5. Any error messages

---

## 🎯 Success Criteria

You'll know it's working when:
✅ Two windows appear immediately
✅ Camera feed shows in debug window
✅ Green skeleton overlays your hand
✅ Gesture name appears (e.g., "PINCH", "OPEN PALM")
✅ 3D window shows colorful cubes
✅ Hand skeleton visible in 3D space
✅ Pinch near cube → cube highlights
✅ Move hand → cube moves with it
✅ Open palm → cube releases

---

**Last Updated:** 2025-11-30
**Version:** 1.1 (with coordinate fix)

