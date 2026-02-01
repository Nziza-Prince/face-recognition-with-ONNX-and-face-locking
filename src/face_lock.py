# src/face_lock.py
"""
Face Locking and Action Detection System

Locks onto a specific enrolled identity and tracks:
- Face movement (left/right)
- Eye blinks
- Smiles/laughs

Records action history to: <name>_history_<timestamp>.txt

Run:
python -m src.face_lock

Keys:
q : quit
r : reload DB
l : lock onto detected face (if recognized as target)
u : unlock current face
+/- : adjust recognition threshold
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    MP_IMPORT_ERROR = e

from .haar_5pt import align_face_5pt
from .recognize import (
    HaarFaceMesh5pt,
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    load_db_npz,
    FaceDet,
)

# ----------------------------------
# Action Detection Data
# ----------------------------------

@dataclass
class ActionRecord:
    timestamp: float
    action_type: str
    description: str
    
    def to_line(self) -> str:
        dt = datetime.fromtimestamp(self.timestamp)
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return f"{time_str} | {self.action_type:15s} | {self.description}"

@dataclass
class FaceState:
    """Tracks state for locked face"""
    center_x: float
    center_y: float
    last_blink_time: float
    last_smile_time: float
    eye_aspect_ratio: float
    mouth_aspect_ratio: float
    consecutive_failures: int

# ----------------------------------
# Action Detector
# ----------------------------------

class ActionDetector:
    """Detects face actions using MediaPipe landmarks"""
    
    def __init__(self):
        if mp is None:
            raise RuntimeError(f"MediaPipe required: {MP_IMPORT_ERROR}")
        
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Eye landmarks (for blink detection)
        self.LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
        
        # Mouth landmarks (for smile detection)
        self.MOUTH_IDXS = [61, 291, 0, 17, 78, 308]
        
        # Thresholds
        self.BLINK_EAR_THRESH = 0.21
        self.SMILE_MAR_THRESH = 0.35
        self.MOVEMENT_THRESH = 30  # pixels
        
    def _eye_aspect_ratio(self, eye_points: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return float(ear)
    
    def _mouth_aspect_ratio(self, mouth_points: np.ndarray) -> float:
        """Calculate Mouth Aspect Ratio (MAR) for smile detection"""
        # Vertical distance (mouth opening)
        v = np.linalg.norm(mouth_points[2] - mouth_points[3])
        # Horizontal distance (mouth width)
        h = np.linalg.norm(mouth_points[0] - mouth_points[1])
        
        mar = v / (h + 1e-6)
        return float(mar)
    
    def detect_actions(
        self,
        frame_bgr: np.ndarray,
        face_roi: Tuple[int, int, int, int],
        prev_state: Optional[FaceState]
    ) -> Tuple[List[str], FaceState]:
        """
        Detect actions in the face ROI.
        Returns: (list of detected actions, updated face state)
        """
        x1, y1, x2, y2 = face_roi
        roi = frame_bgr[y1:y2, x1:x2]
        
        if roi.size == 0:
            if prev_state:
                return [], prev_state
            return [], FaceState(0, 0, 0, 0, 0.3, 0.2, 0)
        
        H, W = roi.shape[:2]
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        
        actions = []
        current_time = time.time()
        
        # Calculate face center in full frame
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        if not res.multi_face_landmarks:
            if prev_state:
                return actions, FaceState(
                    center_x, center_y,
                    prev_state.last_blink_time,
                    prev_state.last_smile_time,
                    prev_state.eye_aspect_ratio,
                    prev_state.mouth_aspect_ratio,
                    prev_state.consecutive_failures + 1
                )
            return actions, FaceState(center_x, center_y, 0, 0, 0.3, 0.2, 1)
        
        lm = res.multi_face_landmarks[0].landmark
        
        # Extract eye landmarks
        left_eye = np.array([[lm[i].x * W, lm[i].y * H] for i in self.LEFT_EYE_IDXS])
        right_eye = np.array([[lm[i].x * W, lm[i].y * H] for i in self.RIGHT_EYE_IDXS])
        
        # Extract mouth landmarks
        mouth = np.array([[lm[i].x * W, lm[i].y * H] for i in self.MOUTH_IDXS])
        
        # Calculate ratios
        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        mar = self._mouth_aspect_ratio(mouth)
        
        # Detect blink
        if prev_state and avg_ear < self.BLINK_EAR_THRESH:
            if prev_state.eye_aspect_ratio >= self.BLINK_EAR_THRESH:
                if current_time - prev_state.last_blink_time > 0.3:
                    actions.append("BLINK")
                    last_blink = current_time
                else:
                    last_blink = prev_state.last_blink_time
            else:
                last_blink = prev_state.last_blink_time if prev_state else 0
        else:
            last_blink = prev_state.last_blink_time if prev_state else 0
        
        # Detect smile
        if mar > self.SMILE_MAR_THRESH:
            if prev_state and prev_state.mouth_aspect_ratio <= self.SMILE_MAR_THRESH:
                if current_time - prev_state.last_smile_time > 1.0:
                    actions.append("SMILE")
                    last_smile = current_time
                else:
                    last_smile = prev_state.last_smile_time
            else:
                last_smile = prev_state.last_smile_time if prev_state else 0
        else:
            last_smile = prev_state.last_smile_time if prev_state else 0
        
        # Detect movement (left/right)
        if prev_state:
            dx = center_x - prev_state.center_x
            if abs(dx) > self.MOVEMENT_THRESH:
                if dx > 0:
                    actions.append("MOVE_RIGHT")
                else:
                    actions.append("MOVE_LEFT")
        
        new_state = FaceState(
            center_x=center_x,
            center_y=center_y,
            last_blink_time=last_blink,
            last_smile_time=last_smile,
            eye_aspect_ratio=avg_ear,
            mouth_aspect_ratio=mar,
            consecutive_failures=0
        )
        
        return actions, new_state

# ----------------------------------
# Face Lock Manager
# ----------------------------------

class FaceLockManager:
    """Manages face locking and action history"""
    
    def __init__(self, target_name: str, history_dir: Path = Path("data/history")):
        self.target_name = target_name
        self.history_dir = history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_locked = False
        self.locked_face: Optional[FaceDet] = None
        self.face_state: Optional[FaceState] = None
        self.action_history: List[ActionRecord] = []
        
        self.max_failures = 15  # frames before unlock
        self.lock_time: Optional[float] = None
        
        # Action detector
        self.action_detector = ActionDetector()
        
        # History file
        self.history_file: Optional[Path] = None
    
    def try_lock(self, face: FaceDet, name: str) -> bool:
        """Try to lock onto a face if it matches target"""
        if name == self.target_name and not self.is_locked:
            self.is_locked = True
            self.locked_face = face
            self.lock_time = time.time()
            self.face_state = None
            
            # Create history file
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.history_file = self.history_dir / f"{self.target_name}_history_{timestamp}.txt"
            
            # Write header
            with open(self.history_file, 'w') as f:
                f.write(f"Face Lock History for: {self.target_name}\n")
                f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")
            
            self._record_action("LOCK", f"Face locked onto {self.target_name}")
            return True
        return False
    
    def unlock(self, reason: str = "Manual unlock"):
        """Unlock the current face"""
        if self.is_locked:
            self._record_action("UNLOCK", reason)
            
            # Write summary
            if self.history_file:
                duration = time.time() - self.lock_time if self.lock_time else 0
                with open(self.history_file, 'a') as f:
                    f.write("\n" + "=" * 70 + "\n")
                    f.write(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Duration: {duration:.1f} seconds\n")
                    f.write(f"Total actions recorded: {len(self.action_history)}\n")
            
            self.is_locked = False
            self.locked_face = None
            self.face_state = None
            self.lock_time = None
    
    def update_tracking(self, frame: np.ndarray, faces: List[FaceDet], matcher, embedder) -> Optional[FaceDet]:
        """Update tracking for locked face"""
        if not self.is_locked:
            return None
        
        # Try to find the locked face
        best_face = None
        best_distance = float('inf')
        
        for face in faces:
            # Check if face is close to last known position
            if self.locked_face:
                dx = abs((face.x1 + face.x2) / 2 - (self.locked_face.x1 + self.locked_face.x2) / 2)
                dy = abs((face.y1 + face.y2) / 2 - (self.locked_face.y1 + self.locked_face.y2) / 2)
                
                # If face moved too far, skip
                if dx > 150 or dy > 150:
                    continue
            
            # Verify identity
            aligned = align_face_5pt(frame, face.kps, out_size=(112, 112))
            emb = embedder.embed(aligned)
            mr = matcher.match(emb)
            
            if mr.name == self.target_name and mr.distance < best_distance:
                best_face = face
                best_distance = mr.distance
        
        if best_face:
            # Update locked face
            self.locked_face = best_face
            
            # Detect actions
            roi = (best_face.x1, best_face.y1, best_face.x2, best_face.y2)
            actions, new_state = self.action_detector.detect_actions(frame, roi, self.face_state)
            self.face_state = new_state
            
            # Record actions
            for action in actions:
                desc = self._get_action_description(action, new_state)
                self._record_action(action, desc)
            
            return best_face
        else:
            # Face lost
            if self.face_state:
                self.face_state.consecutive_failures += 1
                if self.face_state.consecutive_failures >= self.max_failures:
                    self.unlock("Face lost for too long")
            return None
    
    def _get_action_description(self, action: str, state: FaceState) -> str:
        """Generate description for action"""
        if action == "BLINK":
            return f"Eye blink detected (EAR={state.eye_aspect_ratio:.3f})"
        elif action == "SMILE":
            return f"Smile detected (MAR={state.mouth_aspect_ratio:.3f})"
        elif action == "MOVE_LEFT":
            return f"Face moved left (x={state.center_x:.1f})"
        elif action == "MOVE_RIGHT":
            return f"Face moved right (x={state.center_x:.1f})"
        else:
            return action
    
    def _record_action(self, action_type: str, description: str):
        """Record action to history"""
        record = ActionRecord(
            timestamp=time.time(),
            action_type=action_type,
            description=description
        )
        self.action_history.append(record)
        
        # Write to file
        if self.history_file:
            with open(self.history_file, 'a') as f:
                f.write(record.to_line() + "\n")
        
        print(f"[ACTION] {action_type}: {description}")

# ----------------------------------
# Main Demo
# ----------------------------------

def main():
    db_path = Path("data/db/face_db.npz")
    
    # Load database
    db = load_db_npz(db_path)
    if not db:
        print("ERROR: No face database found. Run enrollment first.")
        print("  python -m src.enroll")
        return
    
    print("Available identities:")
    for i, name in enumerate(sorted(db.keys()), 1):
        print(f"  {i}. {name}")
    
    target_name = input("\nEnter name to lock onto: ").strip()
    if target_name not in db:
        print(f"ERROR: '{target_name}' not found in database.")
        return
    
    # Initialize components
    det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False
    )
    matcher = FaceDBMatcher(db=db, dist_thresh=0.34)
    
    # Face lock manager
    lock_mgr = FaceLockManager(target_name=target_name)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")
    
    print(f"\nFace Lock System Started")
    print(f"Target: {target_name}")
    print("Controls:")
    print("  l = lock onto detected face (if target)")
    print("  u = unlock")
    print("  q = quit")
    print("  r = reload DB")
    print("  +/- = adjust threshold\n")
    
    t0 = time.time()
    frames = 0
    fps = 0.0
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            vis = frame.copy()
            faces = det.detect(frame, max_faces=5)
            
            # FPS
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()
            
            # Process faces
            if lock_mgr.is_locked:
                # Track locked face
                locked_face = lock_mgr.update_tracking(frame, faces, matcher, embedder)
                
                if locked_face:
                    # Draw locked face with special highlight
                    cv2.rectangle(vis, (locked_face.x1, locked_face.y1), 
                                (locked_face.x2, locked_face.y2), (0, 255, 255), 3)
                    
                    for (x, y) in locked_face.kps.astype(int):
                        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 255), -1)
                    
                    # Lock indicator
                    cv2.putText(vis, f"LOCKED: {target_name}", 
                              (locked_face.x1, locked_face.y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    
                    # Show state info
                    if lock_mgr.face_state:
                        state = lock_mgr.face_state
                        info = f"EAR:{state.eye_aspect_ratio:.2f} MAR:{state.mouth_aspect_ratio:.2f}"
                        cv2.putText(vis, info, (locked_face.x1, locked_face.y2 + 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # Show all faces, allow locking
                for face in faces:
                    aligned = align_face_5pt(frame, face.kps, out_size=(112, 112))
                    emb = embedder.embed(aligned)
                    mr = matcher.match(emb)
                    
                    label = mr.name if mr.name else "Unknown"
                    color = (0, 255, 0) if mr.accepted else (0, 0, 255)
                    
                    # Highlight target
                    if label == target_name:
                        color = (255, 0, 255)
                        label = f"{label} [TARGET - Press L]"
                    
                    cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), color, 2)
                    for (x, y) in face.kps.astype(int):
                        cv2.circle(vis, (int(x), int(y)), 2, color, -1)
                    
                    cv2.putText(vis, label, (face.x1, face.y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Status overlay
            status_lines = [
                f"Target: {target_name}",
                f"Status: {'LOCKED' if lock_mgr.is_locked else 'SEARCHING'}",
                f"Actions: {len(lock_mgr.action_history)}",
                f"FPS: {fps:.1f}"
            ]
            
            y = 30
            for line in status_lines:
                color = (0, 255, 255) if lock_mgr.is_locked else (255, 255, 255)
                cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, color, 2, cv2.LINE_AA)
                y += 30
            
            cv2.imshow("Face Lock System", vis)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('l'):
                if not lock_mgr.is_locked and faces:
                    # Try to lock onto target face
                    for face in faces:
                        aligned = align_face_5pt(frame, face.kps, out_size=(112, 112))
                        emb = embedder.embed(aligned)
                        mr = matcher.match(emb)
                        if lock_mgr.try_lock(face, mr.name):
                            print(f"Locked onto {mr.name}")
                            break
            elif key == ord('u'):
                if lock_mgr.is_locked:
                    lock_mgr.unlock("Manual unlock")
                    print("Unlocked")
            elif key == ord('r'):
                matcher.reload_from(db_path)
                print(f"Reloaded DB: {len(matcher._names)} identities")
            elif key in (ord('+'), ord('=')):
                matcher.dist_thresh = min(1.0, matcher.dist_thresh + 0.01)
                print(f"Threshold: {matcher.dist_thresh:.2f}")
            elif key == ord('-'):
                matcher.dist_thresh = max(0.05, matcher.dist_thresh - 0.01)
                print(f"Threshold: {matcher.dist_thresh:.2f}")
    
    finally:
        if lock_mgr.is_locked:
            lock_mgr.unlock("Session ended")
        cap.release()
        cv2.destroyAllWindows()
        
        if lock_mgr.history_file:
            print(f"\nAction history saved to: {lock_mgr.history_file}")

if __name__ == "__main__":
    main()
