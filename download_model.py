#!/usr/bin/env python3
"""
Download script for ArcFace ONNX model.
Run this script to automatically download the required model file.
"""

import os
import urllib.request
from pathlib import Path

def download_model():
    """Download the ArcFace ONNX model."""
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "embedder_arcface.onnx"
    
    if model_path.exists():
        print(f"Model already exists at {model_path}")
        file_size = model_path.stat().st_size / (1024 * 1024)  # MB
        print(f"File size: {file_size:.1f} MB")
        
        response = input("Do you want to re-download? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            return
    
    # Model URLs (try multiple sources)
    model_urls = [
        "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
        # Add more backup URLs here if needed
    ]
    
    print("Downloading ArcFace ONNX model...")
    
    for i, url in enumerate(model_urls):
        try:
            print(f"Trying source {i+1}/{len(model_urls)}: {url}")
            
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    print(f"\rProgress: {percent}%", end="", flush=True)
            
            urllib.request.urlretrieve(url, model_path, progress_hook)
            print(f"\nModel downloaded successfully to {model_path}")
            
            # Verify file size
            file_size = model_path.stat().st_size / (1024 * 1024)  # MB
            print(f"File size: {file_size:.1f} MB")
            
            if file_size < 100:  # Expect ~249MB
                print("Warning: File size seems too small. Download might be incomplete.")
            else:
                print("Download completed successfully!")
            return
            
        except Exception as e:
            print(f"\nFailed to download from {url}: {e}")
            if model_path.exists():
                model_path.unlink()  # Remove partial download
            continue
    
    print("\nAll download sources failed. Please try manual download:")
    print("1. Go to: https://github.com/onnx/models/tree/main/vision/body_analysis/arcface")
    print("2. Download the ArcFace ONNX model")
    print(f"3. Save it as: {model_path}")

if __name__ == "__main__":
    download_model()