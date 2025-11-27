#!/usr/bin/env python3
"""
Diagnostic script to test individual components of the Chaplin application
"""

import torch
import cv2
import os
import sys

def test_camera():
    """Test webcam access"""
    print("Testing camera access...")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera cannot be opened")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            print(f"✅ Camera working - captured frame shape: {frame.shape}")
            return True
        else:
            print("❌ Cannot read frame from camera")
            return False
    except Exception as e:
        print(f"❌ Camera error: {e}")
        return False

def test_model_files():
    """Test if model files exist and are accessible"""
    print("\nTesting model files...")
    
    model_files = [
        "benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth",
        "benchmarks/LRS3/models/LRS3_V_WER19.1/model.json",
        "benchmarks/LRS3/language_models/lm_en_subword/model.pth",
        "benchmarks/LRS3/language_models/lm_en_subword/model.json"
    ]
    
    all_good = True
    for file_path in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} exists ({size:,} bytes)")
        else:
            print(f"❌ {file_path} missing")
            all_good = False
    
    return all_good

def test_torch():
    """Test PyTorch installation"""
    print("\nTesting PyTorch...")
    try:
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA device count: {torch.cuda.device_count()}")
        return True
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False

def test_ollama():
    """Test Ollama connection"""
    print("\nTesting Ollama...")
    try:
        from ollama import Client
        client = Client()
        
        # Try to list models
        models = client.list()
        print(f"✅ Ollama connection successful")
        print(f"✅ Available models: {len(models.get('models', []))}")
        
        # Check for qwen3:4b specifically
        has_qwen = any('qwen3:4b' in str(model) for model in models.get('models', []))
        if has_qwen:
            print("✅ qwen3:4b model found")
        else:
            print("⚠️  qwen3:4b model not found - will need to download")
        
        return True
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        print("💡 Make sure Ollama server is running: `ollama serve`")
        return False

def main():
    print("=== Chaplin Application Diagnostic ===\n")
    
    tests = [
        ("Camera", test_camera),
        ("Model Files", test_model_files), 
        ("PyTorch", test_torch),
        ("Ollama", test_ollama)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results[name] = False
    
    print("\n=== Summary ===")
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
    
    # Return exit code based on critical components
    if not results["Camera"] or not results["Model Files"] or not results["PyTorch"]:
        print("\n🔧 Critical issues found - application will not work properly")
        sys.exit(1)
    elif not results["Ollama"]:
        print("\n⚠️  Ollama issues found - text correction will not work")
        print("💡 You can still run the app but text correction will be disabled")
        sys.exit(0)
    else:
        print("\n🎉 All tests passed - application should work!")
        sys.exit(0)

if __name__ == "__main__":
    main()