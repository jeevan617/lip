#!/usr/bin/env python3
"""
Test 5-point landmark processing
"""
import numpy as np
import sys
import os
import cv2

sys.path.insert(0, os.getcwd())

from pipelines.detectors.retinaface.video_process import VideoProcess

# Create a dummy video (20 frames)
video = np.random.randint(0, 255, (20, 480, 640, 3), dtype=np.uint8)

# Create dummy landmarks for 20 frames (5 points each)
# Points: Left Eye, Right Eye, Nose, Left Mouth, Right Mouth
landmarks = []
for _ in range(20):
    lm = np.array([
        [200, 200], # Left Eye
        [300, 200], # Right Eye
        [250, 250], # Nose
        [220, 300], # Left Mouth
        [280, 300]  # Right Mouth
    ], dtype=np.float32)
    landmarks.append(lm)

print(f"Video shape: {video.shape}")
print(f"Landmarks count: {len(landmarks)}")
print(f"Landmarks shape: {landmarks[0].shape}")

# Initialize processor
processor = VideoProcess()

try:
    # This exercises the affine_transform and get_stable_reference logic
    result = processor(video, landmarks)
    if result is not None:
        print(f"✅ Success! Result shape: {result.shape}")
    else:
        print("❌ Result is None")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
