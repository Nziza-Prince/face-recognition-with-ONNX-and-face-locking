#!/usr/bin/env python3
"""
Guide for re-enrolling identities with better quality.
"""

print("""
╔══════════════════════════════════════════════════════════════════════╗
║              RE-ENROLLMENT GUIDE FOR BETTER ACCURACY                 ║
╚══════════════════════════════════════════════════════════════════════╝

PROBLEM: Your identities are too similar (distance: 0.006, need: >0.30)

SOLUTION: Re-enroll with these best practices:

1. LIGHTING
   ✓ Use bright, even lighting (avoid shadows on face)
   ✓ Face the light source directly
   ✗ Avoid backlighting (light behind you)
   ✗ Avoid harsh side lighting

2. CAMERA POSITION
   ✓ Face camera directly
   ✓ Keep face centered in frame
   ✓ Maintain consistent distance (arm's length)
   ✗ Avoid extreme angles

3. CAPTURE MORE SAMPLES
   ✓ Capture 25-30 samples (instead of 15)
   ✓ Vary your expression slightly:
     - Neutral face
     - Slight smile
     - Serious expression
   ✓ Vary head angle slightly:
     - Look straight
     - Turn head slightly left (5-10°)
     - Turn head slightly right (5-10°)
     - Tilt head slightly up/down (5°)

4. DURING ENROLLMENT
   ✓ Use auto-capture mode (press 'a')
   ✓ Move slowly and naturally
   ✓ Keep eyes open and looking at camera
   ✗ Don't make extreme expressions
   ✗ Don't move too fast

5. ENVIRONMENT
   ✓ Clean background (plain wall is best)
   ✓ Remove glasses if possible (or capture with/without)
   ✓ Remove hats/caps
   ✗ Avoid busy/cluttered backgrounds

═══════════════════════════════════════════════════════════════════════

STEP-BY-STEP RE-ENROLLMENT:

1. Delete existing enrollment data:
   
   rm -rf data/enroll/Nziza
   rm -rf data/enroll/Dios

2. Re-enroll first person:
   
   python -m src.enroll
   
   - Enter name: Nziza
   - Press 'a' for auto-capture
   - Slowly turn head left and right
   - Vary expressions slightly
   - Wait for 25-30 samples
   - Press 's' to save

3. Re-enroll second person:
   
   python -m src.enroll
   
   - Enter name: Dios
   - Press 'a' for auto-capture
   - Follow same process
   - Press 's' to save

4. Verify improvement:
   
   python diagnose_recognition.py
   
   - Check distance between identities
   - Should be > 0.30 for good separation

5. Test recognition:
   
   python -m src.recognize
   
   - Test with both people
   - Verify correct labels

═══════════════════════════════════════════════════════════════════════

QUICK COMMANDS:

# Delete old data
rm -rf data/enroll/Nziza data/enroll/Dios

# Re-enroll (run twice, once for each person)
python -m src.enroll

# Check if it worked
python diagnose_recognition.py

# Test it
python -m src.recognize

═══════════════════════════════════════════════════════════════════════
""")
