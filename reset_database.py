#!/usr/bin/env python3
"""
Reset face database - removes all enrolled identities.
Use this to start fresh with better enrollment quality.
"""

import sys
from pathlib import Path
import shutil

def reset_database():
    """Reset the face database"""
    
    print("=" * 70)
    print("Face Database Reset Tool")
    print("=" * 70)
    print()
    
    # Paths
    db_npz = Path("data/db/face_db.npz")
    db_json = Path("data/db/face_db.json")
    enroll_dir = Path("data/enroll")
    
    # Check what exists
    has_db = db_npz.exists()
    has_json = db_json.exists()
    has_enroll = enroll_dir.exists() and any(enroll_dir.iterdir())
    
    if not (has_db or has_json or has_enroll):
        print("✓ Database is already empty. Nothing to reset.")
        return
    
    print("Current database contents:")
    
    if has_db:
        import numpy as np
        data = np.load(db_npz)
        identities = list(data.files)
        print(f"  - Face database: {len(identities)} identities")
        for name in identities:
            print(f"    • {name}")
    
    if has_enroll:
        enroll_dirs = [d for d in enroll_dir.iterdir() if d.is_dir()]
        print(f"  - Enrollment data: {len(enroll_dirs)} identities")
        for d in enroll_dirs:
            num_files = len(list(d.glob("*.jpg")))
            print(f"    • {d.name}: {num_files} samples")
    
    print()
    print("⚠️  WARNING: This will DELETE all enrolled identities!")
    print()
    
    response = input("Are you sure you want to reset? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\nReset cancelled.")
        return
    
    print()
    print("Resetting database...")
    
    # Remove database files
    if db_npz.exists():
        db_npz.unlink()
        print("  ✓ Removed face_db.npz")
    
    if db_json.exists():
        db_json.unlink()
        print("  ✓ Removed face_db.json")
    
    # Remove enrollment directories
    if enroll_dir.exists():
        for item in enroll_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                print(f"  ✓ Removed enrollment data for {item.name}")
    
    print()
    print("=" * 70)
    print("✓ Database reset complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Re-enroll identities with better quality:")
    print("     python -m src.enroll")
    print()
    print("  2. Follow best practices:")
    print("     - Use good lighting")
    print("     - Capture 25-30 samples")
    print("     - Vary head angles and expressions")
    print("     - Use auto-capture mode (press 'a')")
    print()
    print("  3. Verify improvement:")
    print("     python diagnose_recognition.py")
    print()

def main():
    try:
        reset_database()
    except KeyboardInterrupt:
        print("\n\nReset cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
