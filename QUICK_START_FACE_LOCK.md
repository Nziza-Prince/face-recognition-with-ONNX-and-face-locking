# Face Lock - Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies
```bash
pip install opencv-python numpy onnxruntime mediapipe
```

### Step 2: Download Model
```bash
python download_model.py
```

### Step 3: Enroll a Face
```bash
python -m src.enroll
```
- Enter a name (e.g., "Gabi")
- Press **SPACE** to capture samples (need ~15)
- Press **S** to save
- Press **Q** to quit

### Step 4: Run Face Lock
```bash
python -m src.face_lock
```
- Enter the name you enrolled
- Press **L** when your face appears to lock
- Move around, blink, smile - actions will be recorded!
- Press **Q** to quit

### Step 5: View Results
```bash
cat data/history/*_history_*.txt
```

## 🎯 What It Does

### Detects 4 Actions:
1. **👈 MOVE_LEFT** - Face moves left
2. **👉 MOVE_RIGHT** - Face moves right  
3. **👁️ BLINK** - Eyes close
4. **😊 SMILE** - Mouth opens in smile

### Records Everything:
```
2026-01-29 11:20:45.123 | LOCK       | Face locked onto Gabi
2026-01-29 11:20:47.456 | BLINK      | Eye blink detected (EAR=0.189)
2026-01-29 11:20:49.789 | SMILE      | Smile detected (MAR=0.412)
2026-01-29 11:20:51.234 | MOVE_LEFT  | Face moved left (x=245.3)
```

## ⌨️ Controls

| Key | Action |
|-----|--------|
| **L** | Lock onto face |
| **U** | Unlock face |
| **Q** | Quit |
| **R** | Reload database |
| **+/-** | Adjust threshold |

## 🧪 Test Everything Works

```bash
python test_face_lock.py
```

Should show all ✓ checks passing.

## 📁 Output Files

Action history saved to:
```
data/history/<name>_history_<timestamp>.txt
```

Example:
```
data/history/gabi_history_20260129112099.txt
```

## 🔧 Troubleshooting

### "No face database found"
```bash
python -m src.enroll
```

### "Model not found"
```bash
python download_model.py
```

### "Camera not available"
- Check camera permissions
- Try different camera index (edit code: `cv2.VideoCapture(1)`)

### Actions not detecting
- Ensure good lighting
- Make actions more pronounced
- Face camera directly

## 📚 More Info

- **Full Documentation**: `FACE_LOCK_EXAMPLE.md`
- **Implementation Details**: `FACE_LOCK_IMPLEMENTATION.md`
- **Main README**: `README.md`

## 🎓 Example Session

```bash
# 1. Enroll yourself
$ python -m src.enroll
Enter person name: Gabi
[Capture 15 samples with SPACE, then press S]

# 2. Run face lock
$ python -m src.face_lock
Enter name to lock onto: Gabi
[Press L when face appears]

# 3. Perform actions
# - Move your head left and right
# - Blink a few times
# - Smile!

# 4. Check results
$ cat data/history/gabi_history_*.txt
Face Lock History for: Gabi
Session started: 2026-01-29 11:20:45
======================================================================

2026-01-29 11:20:45.123 | LOCK       | Face locked onto Gabi
2026-01-29 11:20:47.456 | BLINK      | Eye blink detected
2026-01-29 11:20:49.789 | SMILE      | Smile detected
2026-01-29 11:20:51.234 | MOVE_LEFT  | Face moved left
2026-01-29 11:20:52.567 | MOVE_RIGHT | Face moved right
...
```

## ✅ Requirements Met

- ✅ Manual face selection
- ✅ Face locking with clear indication
- ✅ Stable tracking across frames
- ✅ Action detection (left, right, blink, smile)
- ✅ Action history recording
- ✅ Correct file naming format
- ✅ CPU-only execution
- ✅ No model retraining
- ✅ Original pipeline intact

## 🎉 That's It!

You now have a working face lock system that:
- Tracks specific people
- Detects their actions
- Records everything to timestamped files

Enjoy! 🚀
