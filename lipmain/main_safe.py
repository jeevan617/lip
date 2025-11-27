#!/usr/bin/env python3
"""
Safe version of main.py with better error handling and Ollama fallback
"""

import torch
import hydra
from pipelines.pipeline import InferencePipeline
from chaplin_safe import Chaplin
import sys


@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    print("=== Chaplin Safe Mode ===")
    
    # Test Ollama availability first
    use_ollama = True
    try:
        from ollama import Client
        client = Client()
        client.list()  # This will fail if Ollama is not running
        print("✅ Ollama server detected - text correction enabled")
    except Exception as e:
        print(f"⚠️  Ollama not available: {e}")
        print("💡 Running without text correction - raw output will be typed")
        use_ollama = False
    
    try:
        chaplin = Chaplin(use_ollama=use_ollama)

        # load the model
        print("\n🔄 Loading VSR model...")
        device = torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() and cfg.gpu_idx >= 0 else "cpu")
        print(f"📱 Using device: {device}")
        
        chaplin.vsr_model = InferencePipeline(
            cfg.config_filename, 
            device=device, 
            detector=cfg.detector, 
            face_track=True
        )

        print("\n\033[48;5;22m\033[97m\033[1m MODEL LOADED SUCCESSFULLY! \033[0m\n")

        # start the webcam video capture
        chaplin.start_webcam()
        
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("  - Make sure your webcam is connected and accessible")
        print("  - Check that all model files are present")
        print("  - Try running with sudo if you have permission issues")
        print("  - Run 'python diagnose.py' for detailed diagnostics")
        sys.exit(1)


if __name__ == '__main__':
    main()