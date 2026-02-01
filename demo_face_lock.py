#!/usr/bin/env python3
"""
Quick demo script for Face Locking feature.

This demonstrates:
1. Loading enrolled identities
2. Locking onto a specific face
3. Tracking actions (movement, blinks, smiles)
4. Recording action history

Usage:
    python demo_face_lock.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.face_lock import main

if __name__ == "__main__":
    print("=" * 70)
    print("Face Locking and Action Detection Demo")
    print("=" * 70)
    print()
    print("This demo will:")
    print("  1. Show all enrolled identities")
    print("  2. Let you select one to lock onto")
    print("  3. Track that face and detect actions:")
    print("     - Face movement (left/right)")
    print("     - Eye blinks")
    print("     - Smiles/laughs")
    print("  4. Save action history to a timestamped file")
    print()
    print("Make sure you have:")
    print("  - Enrolled at least one identity (python -m src.enroll)")
    print("  - Camera connected and working")
    print()
    print("=" * 70)
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        print("\nTroubleshooting:")
        print("  - Run enrollment first: python -m src.enroll")
        print("  - Check camera connection")
        print("  - Ensure MediaPipe is installed: pip install mediapipe")
