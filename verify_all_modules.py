#!/usr/bin/env python3
"""
Verify all modules can be imported and initialized without errors.
"""

import sys

def test_module_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        from src import detect, landmarks, align, embed, enroll, recognize, face_lock
        print("  ✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_align_face_5pt():
    """Test align_face_5pt returns correct tuple"""
    print("\nTesting align_face_5pt function...")
    
    try:
        import numpy as np
        from src.haar_5pt import align_face_5pt
        
        # Create dummy data
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kps = np.array([
            [200, 200],  # left eye
            [300, 200],  # right eye
            [250, 250],  # nose
            [220, 300],  # left mouth
            [280, 300],  # right mouth
        ], dtype=np.float32)
        
        # Test function
        result = align_face_5pt(frame, kps, out_size=(112, 112))
        
        # Verify it returns a tuple
        if not isinstance(result, tuple):
            print(f"  ✗ Expected tuple, got {type(result)}")
            return False
        
        if len(result) != 2:
            print(f"  ✗ Expected tuple of length 2, got {len(result)}")
            return False
        
        aligned, M = result
        
        # Verify aligned image shape
        if aligned.shape != (112, 112, 3):
            print(f"  ✗ Expected aligned shape (112, 112, 3), got {aligned.shape}")
            return False
        
        # Verify transformation matrix shape
        if M.shape != (2, 3):
            print(f"  ✗ Expected M shape (2, 3), got {M.shape}")
            return False
        
        print("  ✓ align_face_5pt returns correct tuple (aligned, M)")
        print(f"    - aligned shape: {aligned.shape}")
        print(f"    - M shape: {M.shape}")
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedder():
    """Test embedder can process aligned face"""
    print("\nTesting ArcFace embedder...")
    
    try:
        import numpy as np
        from src.recognize import ArcFaceEmbedderONNX
        
        embedder = ArcFaceEmbedderONNX(
            model_path="models/embedder_arcface.onnx",
            input_size=(112, 112),
            debug=False
        )
        
        # Create dummy aligned face
        aligned = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        # Test embedding
        emb = embedder.embed(aligned)
        
        # Verify embedding
        if not isinstance(emb, np.ndarray):
            print(f"  ✗ Expected ndarray, got {type(emb)}")
            return False
        
        if emb.ndim != 1:
            print(f"  ✗ Expected 1D array, got {emb.ndim}D")
            return False
        
        # Check if normalized
        norm = np.linalg.norm(emb)
        if not (0.99 < norm < 1.01):
            print(f"  ✗ Expected L2-normalized (norm≈1.0), got norm={norm:.3f}")
            return False
        
        print("  ✓ Embedder works correctly")
        print(f"    - embedding dim: {emb.shape[0]}")
        print(f"    - L2 norm: {norm:.6f}")
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
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
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("Module Verification Test")
    print("=" * 70)
    print()
    
    results = []
    
    results.append(("Module Imports", test_module_imports()))
    results.append(("align_face_5pt", test_align_face_5pt()))
    results.append(("ArcFace Embedder", test_embedder()))
    results.append(("ActionDetector", test_action_detector()))
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:25s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("All verification tests passed! ✓")
        print("\nAll modules are working correctly:")
        print("  - src.recognize (multi-face recognition)")
        print("  - src.enroll (enrollment system)")
        print("  - src.embed (embedding generation)")
        print("  - src.face_lock (face locking & action detection)")
    else:
        print("Some tests failed. ✗")
        print("\nPlease check the errors above.")
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
