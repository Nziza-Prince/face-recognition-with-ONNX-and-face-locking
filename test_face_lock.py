#!/usr/bin/env python3
"""
Test script for Face Lock feature.
Verifies that all components are working correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        import cv2
        print("  ✓ OpenCV")
    except ImportError as e:
        print(f"  ✗ OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print("  ✓ NumPy")
    except ImportError as e:
        print(f"  ✗ NumPy: {e}")
        return False
    
    try:
        import mediapipe as mp
        print(f"  ✓ MediaPipe (version: {mp.__version__})")
    except ImportError as e:
        print(f"  ✗ MediaPipe: {e}")
        return False
    
    try:
        import onnxruntime as ort
        print(f"  ✓ ONNX Runtime (version: {ort.__version__})")
    except ImportError as e:
        print(f"  ✗ ONNX Runtime: {e}")
        return False
    
    return True

def test_modules():
    """Test that face lock modules can be imported"""
    print("\nTesting face lock modules...")
    try:
        from src.face_lock import (
            ActionDetector,
            FaceLockManager,
            ActionRecord,
            FaceState
        )
        print("  ✓ Face lock components")
    except ImportError as e:
        print(f"  ✗ Face lock components: {e}")
        return False
    
    try:
        from src.recognize import (
            HaarFaceMesh5pt,
            ArcFaceEmbedderONNX,
            FaceDBMatcher
        )
        print("  ✓ Recognition components")
    except ImportError as e:
        print(f"  ✗ Recognition components: {e}")
        return False
    
    return True

def test_model():
    """Test that the ArcFace model exists"""
    print("\nTesting model file...")
    model_path = Path("models/embedder_arcface.onnx")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ ArcFace model found ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ✗ ArcFace model not found at {model_path}")
        print("    Run: python download_model.py")
        return False

def test_database():
    """Test that face database exists"""
    print("\nTesting face database...")
    db_path = Path("data/db/face_db.npz")
    if db_path.exists():
        import numpy as np
        data = np.load(db_path, allow_pickle=True)
        identities = list(data.files)
        print(f"  ✓ Face database found with {len(identities)} identities:")
        for name in identities:
            print(f"      - {name}")
        return True
    else:
        print(f"  ✗ Face database not found at {db_path}")
        print("    Run: python -m src.enroll")
        return False

def test_camera():
    """Test that camera is accessible"""
    print("\nTesting camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"  ✓ Camera accessible (frame: {frame.shape})")
                return True
            else:
                print("  ✗ Camera opened but cannot read frames")
                return False
        else:
            print("  ✗ Cannot open camera")
            return False
    except Exception as e:
        print(f"  ✗ Camera test failed: {e}")
        return False

def test_action_detector():
    """Test ActionDetector initialization"""
    print("\nTesting ActionDetector...")
    try:
        from src.face_lock import ActionDetector
        detector = ActionDetector()
        print("  ✓ ActionDetector initialized")
        print(f"    - Blink threshold: {detector.BLINK_EAR_THRESH}")
        print(f"    - Smile threshold: {detector.SMILE_MAR_THRESH}")
        print(f"    - Movement threshold: {detector.MOVEMENT_THRESH}")
        return True
    except Exception as e:
        print(f"  ✗ ActionDetector failed: {e}")
        return False

def main():
    print("=" * 70)
    print("Face Lock System - Component Test")
    print("=" * 70)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Modules", test_modules()))
    results.append(("Model", test_model()))
    results.append(("Database", test_database()))
    results.append(("Camera", test_camera()))
    results.append(("ActionDetector", test_action_detector()))
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("All tests passed! ✓")
        print("\nYou can now run the face lock system:")
        print("  python -m src.face_lock")
        print("  or")
        print("  python demo_face_lock.py")
    else:
        print("Some tests failed. ✗")
        print("\nPlease fix the issues above before running face lock.")
        print("\nQuick fixes:")
        print("  - Install dependencies: pip install opencv-python numpy onnxruntime mediapipe")
        print("  - Download model: python download_model.py")
        print("  - Enroll faces: python -m src.enroll")
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
