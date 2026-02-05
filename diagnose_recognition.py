#!/usr/bin/env python3
"""
Diagnostic tool for face recognition issues.
Analyzes embeddings and similarity scores to help debug recognition problems.
"""

import numpy as np
from pathlib import Path
import json

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return float(np.dot(a, b))

def cosine_distance(a, b):
    """Calculate cosine distance (1 - similarity)"""
    return 1.0 - cosine_similarity(a, b)

def analyze_database():
    """Analyze the face database"""
    db_path = Path("data/db/face_db.npz")
    json_path = Path("data/db/face_db.json")
    
    if not db_path.exists():
        print("ERROR: No face database found at", db_path)
        return
    
    # Load database
    data = np.load(db_path)
    identities = list(data.files)
    
    print("=" * 70)
    print("Face Database Analysis")
    print("=" * 70)
    print()
    
    # Basic info
    print(f"Total identities: {len(identities)}")
    print(f"Identities: {', '.join(identities)}")
    print()
    
    # Load metadata if available
    if json_path.exists():
        with open(json_path, 'r') as f:
            meta = json.load(f)
        print("Database Metadata:")
        print(f"  Last updated: {meta.get('updated_at', 'N/A')}")
        print(f"  Embedding dimension: {meta.get('embedding_dim', 'N/A')}")
        print()
    
    # Analyze each embedding
    print("Embedding Analysis:")
    print("-" * 70)
    embeddings = {}
    for name in identities:
        emb = data[name].astype(np.float32).reshape(-1)
        embeddings[name] = emb
        
        norm = np.linalg.norm(emb)
        mean_val = np.mean(emb)
        std_val = np.std(emb)
        min_val = np.min(emb)
        max_val = np.max(emb)
        
        print(f"{name}:")
        print(f"  Shape: {emb.shape}")
        print(f"  L2 Norm: {norm:.6f} {'✓' if 0.99 < norm < 1.01 else '✗ NOT NORMALIZED!'}")
        print(f"  Mean: {mean_val:.6f}")
        print(f"  Std Dev: {std_val:.6f}")
        print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
        print()
    
    # Similarity matrix
    if len(identities) > 1:
        print("Similarity Matrix (Cosine Similarity):")
        print("-" * 70)
        print(f"{'':15s}", end="")
        for name in identities:
            print(f"{name:15s}", end="")
        print()
        
        for name1 in identities:
            print(f"{name1:15s}", end="")
            for name2 in identities:
                sim = cosine_similarity(embeddings[name1], embeddings[name2])
                print(f"{sim:15.6f}", end="")
            print()
        print()
        
        # Distance matrix
        print("Distance Matrix (Cosine Distance = 1 - Similarity):")
        print("-" * 70)
        print(f"{'':15s}", end="")
        for name in identities:
            print(f"{name:15s}", end="")
        print()
        
        for name1 in identities:
            print(f"{name1:15s}", end="")
            for name2 in identities:
                dist = cosine_distance(embeddings[name1], embeddings[name2])
                print(f"{dist:15.6f}", end="")
            print()
        print()
        
        # Analysis
        print("Inter-Identity Distance Analysis:")
        print("-" * 70)
        
        distances = []
        for i, name1 in enumerate(identities):
            for name2 in identities[i+1:]:
                dist = cosine_distance(embeddings[name1], embeddings[name2])
                distances.append((name1, name2, dist))
        
        distances.sort(key=lambda x: x[2])
        
        for name1, name2, dist in distances:
            status = "✓ GOOD" if dist > 0.30 else "⚠ TOO SIMILAR" if dist > 0.20 else "✗ VERY SIMILAR"
            print(f"  {name1} ↔ {name2}: {dist:.6f} {status}")
        
        print()
        print("Interpretation:")
        print("  - Distance > 0.40: Very different (excellent separation)")
        print("  - Distance 0.30-0.40: Different (good separation)")
        print("  - Distance 0.20-0.30: Similar (may cause confusion)")
        print("  - Distance < 0.20: Very similar (will cause confusion)")
        print()
    
    # Recommendations
    print("=" * 70)
    print("Recommendations")
    print("=" * 70)
    
    # Check if embeddings are normalized
    all_normalized = all(0.99 < np.linalg.norm(embeddings[n]) < 1.01 for n in identities)
    if not all_normalized:
        print("⚠ WARNING: Some embeddings are not L2-normalized!")
        print("  This can cause incorrect distance calculations.")
        print("  Re-enroll affected identities.")
        print()
    
    # Check similarity
    if len(identities) > 1:
        min_dist = min(d for _, _, d in distances)
        
        if min_dist < 0.20:
            print("✗ CRITICAL: Identities are too similar!")
            print("  The embeddings are very close, causing confusion.")
            print()
            print("  Possible causes:")
            print("  1. Same person enrolled twice with different names")
            print("  2. Very similar-looking people")
            print("  3. Poor quality enrollment samples")
            print()
            print("  Solutions:")
            print("  1. Re-enroll with better lighting and more varied poses")
            print("  2. Capture more samples (20-30 instead of 15)")
            print("  3. Ensure faces are clearly visible and well-lit")
            print("  4. Vary expressions and slight head angles during enrollment")
            print()
        elif min_dist < 0.30:
            print("⚠ WARNING: Some identities are quite similar")
            print("  Recognition may occasionally confuse these people.")
            print()
            print("  Suggestions:")
            print("  1. Lower the recognition threshold (current: 0.34)")
            print("  2. Re-enroll with more diverse samples")
            print("  3. Ensure good lighting during enrollment")
            print()
        else:
            print("✓ GOOD: Identities are well-separated")
            print("  Recognition should work reliably.")
            print()
            print("  Current threshold: 0.34 (distance)")
            print("  Recommended range: 0.30-0.40")
            print()
    
    # Threshold recommendations
    if len(identities) > 1:
        min_dist = min(d for _, _, d in distances)
        
        print("Threshold Recommendations:")
        print(f"  Current threshold: 0.34")
        print(f"  Minimum inter-identity distance: {min_dist:.6f}")
        print()
        
        if min_dist > 0.40:
            print(f"  ✓ You can use threshold up to {min_dist * 0.8:.2f}")
            print(f"    Recommended: 0.35-0.40 for good balance")
        elif min_dist > 0.30:
            print(f"  ⚠ Keep threshold below {min_dist * 0.9:.2f}")
            print(f"    Recommended: 0.28-0.32 to avoid confusion")
        else:
            print(f"  ✗ Threshold must be below {min_dist * 0.9:.2f}")
            print(f"    Recommended: {min_dist * 0.7:.2f}-{min_dist * 0.85:.2f}")
            print(f"    But re-enrollment is strongly recommended!")

def main():
    analyze_database()

if __name__ == "__main__":
    main()
