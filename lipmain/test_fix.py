#!/usr/bin/env python3
"""
Quick test to verify the fix works
"""
import numpy as np
import sys
import os

# Test the video processor with mismatched landmarks
sys.path.insert(0, os.getcwd())

from pipelines.detectors.retinaface.video_process import VideoProcess
from pipelines.data.transforms import VideoTransform
import torch

# Create a dummy video (65 frames, 480x640x3)
video = np.random.randint(0, 255, (65, 480, 640, 3), dtype=np.uint8)

# Create dummy landmarks for only 5 frames (simulating failed detection)
# RetinaFace outputs 5 landmarks per face
landmarks = [np.random.rand(5, 2) * 100 for _ in range(5)]

print(f"Video shape: {video.shape}")
print(f"Landmarks count: {len(landmarks)}")

# Initialize processor
processor = VideoProcess()
transform = VideoTransform(speed_rate=1.0)

# This should NOT crash anymore
try:
    result = processor(video, landmarks)
    if result is not None:
        print(f"✅ Processor Result shape: {result.shape}")
        
        # Convert to tensor like AVSRDataLoader does
        tensor_result = torch.tensor(result)
        print(f"Tensor shape: {tensor_result.shape}")
        
        # Apply transform
        final_result = transform(tensor_result)
        print(f"✅ Transform Result shape: {final_result.shape}")
    else:
        print("❌ Result is None")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
