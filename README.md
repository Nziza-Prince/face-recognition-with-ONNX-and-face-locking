# Face Recognition with ArcFace ONNX

## Description

This project implements a face recognition system using the ArcFace model with ONNX runtime for efficient inference on CPU. It supports multi-face detection, facial landmark detection, face alignment, embedding generation, enrollment, and real-time recognition.

The system uses a pipeline of:

- Haar cascade for multi-face detection
- MediaPipe FaceMesh for 5-point facial landmarks
- Custom alignment to 112x112 resolution
- ArcFace ONNX model for embedding generation
- Cosine distance for face matching

## Features

- **Multi-face Detection**: Detects multiple faces in a frame using Haar cascades
- **Facial Landmark Detection**: Uses MediaPipe FaceMesh for accurate 5-point landmarks
- **Face Alignment**: Aligns faces to a standard 112x112 resolution for consistent embedding
- **Embedding Generation**: Generates face embeddings using the ArcFace ONNX model
- **Enrollment**: Enroll new faces with multiple samples for robust recognition
- **Real-time Recognition**: Recognize faces from camera feed with adjustable threshold
- **Database Management**: Stores face templates in NPZ format with metadata
- **Evaluation Tools**: Evaluate recognition accuracy on datasets

## Installation

1. **Clone or Download**: Ensure you have the project files in your workspace.

2. **Setup Project Structure**:

   ```bash
   python init_project.py
   ```

   This creates the necessary directories and touches required files.

3. **Install Dependencies**:

   ```bash
   pip install opencv-python numpy onnxruntime mediapipe
   ```

   - `opencv-python`: For image processing and Haar cascades
   - `numpy`: For numerical operations
   - `onnxruntime`: For running ONNX models
   - `mediapipe`: For facial landmark detection (optional, but recommended)

4. **Download the ArcFace Model**:

   The project requires the ArcFace ONNX model which is not included in the repository due to size constraints. Download it using one of these methods:

   **Option A: Automatic Download (Recommended)**
   ```bash
   python download_model.py
   ```

   **Option B: Manual Download**
   ```bash
   # Create models directory if it doesn't exist
   mkdir -p models
   
   # Download the ArcFace model
   wget -O models/embedder_arcface.onnx "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"
   ```

   **Option C: Browser Download**
   1. Download the ArcFace ONNX model from: [ONNX Model Zoo](https://github.com/onnx/models/tree/main/vision/body_analysis/arcface)
   2. Save it as `models/embedder_arcface.onnx`

   **Verify Model Installation**:
   ```bash
   ls -la models/embedder_arcface.onnx
   ```
   The file should be approximately 249MB in size.

## Usage

### Enrollment

To enroll new faces into the database:

```bash
python -m src.enroll
```

**Controls**:

- **SPACE**: Capture a single face sample
- **a**: Toggle auto-capture mode
- **s**: Save enrollment after capturing samples
- **r**: Reset new samples (keeps existing crops)
- **q**: Quit

The system will detect faces, align them, generate embeddings, and store templates in `data/db/face_db.npz`.

### Recognition

To run real-time face recognition from camera:

```bash
python -m src.recognize
```

**Controls**:

- **q**: Quit
- **r**: Reload database from disk
- **+/-**: Adjust recognition threshold
- **d**: Toggle debug overlay

The system displays recognized faces with labels and confidence scores.

### Evaluation

To evaluate the system on a dataset:

```bash
python -m src.evaluate
```

This module can be used to test accuracy on prepared datasets.

## Project Structure

```
face_recognition_arcface_onnx/
├── init_project.py          # Project setup script
├── README.md                # This file
├── book/                    # Documentation or notes
├── data/
│   ├── db/                  # Face database (face_db.npz, face_db.json)
│   ├── debug_aligned/       # Debug aligned face images
│   └── enroll/              # Enrollment data per identity
├── models/
│   └── embedder_arcface.onnx  # Pre-trained ArcFace model
└── src/
    ├── align.py             # Face alignment utilities
    ├── camera.py            # Camera interface
    ├── detect.py            # Face detection
    ├── embed.py             # Embedding generation
    ├── enroll.py            # Enrollment module
    ├── evaluate.py          # Evaluation tools
    ├── haar_5pt.py          # Haar cascade with 5-point alignment
    ├── landmarks.py         # Facial landmark detection
    ├── recognize.py         # Real-time recognition
    └── __pycache__/         # Python cache
```

## Requirements

- **Python**: 3.7+
- **OpenCV**: 4.5+
- **NumPy**: 1.19+
- **ONNX Runtime**: 1.8+
- **MediaPipe**: 0.8+ (optional)

## Notes

- The system is optimized for CPU usage.
- FaceMesh is run on individual face ROIs for multi-face support.
- Embeddings are L2-normalized, using cosine distance for matching.
- Database is stored in NumPy's NPZ format for efficiency.

## Troubleshooting

- **Model Not Found Error**: Ensure `models/embedder_arcface.onnx` exists and is approximately 249MB. Re-download if corrupted.
- **MediaPipe Import Error**: Install MediaPipe with `pip install mediapipe`. If issues persist, try `pip install mediapipe==0.10.9`.
- **Camera Permission Issues**: Ensure camera permissions are granted for recognition.
- **ONNX Runtime Error**: Make sure you have the correct ONNX model format. The model should be compatible with ONNX Runtime.
- **Performance Issues**: The system is optimized for CPU. For better performance, ensure good lighting and face the camera directly.
- **Recognition Threshold**: Adjust recognition threshold based on your use case (lower = more sensitive, higher = more strict).

## License

[Add license information if applicable]
