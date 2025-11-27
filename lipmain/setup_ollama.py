#!/usr/bin/env python3
"""
Setup script for Ollama integration
"""

import subprocess
import time
import sys
import os

def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def is_ollama_running():
    """Check if Ollama server is running"""
    try:
        from ollama import Client
        client = Client()
        client.list()
        return True
    except:
        return False

def start_ollama_server():
    """Start Ollama server in background"""
    print("🔄 Starting Ollama server...")
    try:
        # Start Ollama server in background
        process = subprocess.Popen(['ollama', 'serve'], 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL)
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if it's running
        if is_ollama_running():
            print("✅ Ollama server started successfully")
            return True
        else:
            print("❌ Failed to start Ollama server")
            return False
    except Exception as e:
        print(f"❌ Error starting Ollama server: {e}")
        return False

def download_qwen_model():
    """Download the qwen3:4b model"""
    print("🔄 Downloading qwen3:4b model...")
    try:
        result = subprocess.run(['ollama', 'pull', 'qwen3:4b'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ qwen3:4b model downloaded successfully")
            return True
        else:
            print(f"❌ Failed to download model: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def main():
    print("=== Ollama Setup for Chaplin ===\n")
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("❌ Ollama is not installed")
        print("💡 Please install Ollama first:")
        print("   - Visit: https://ollama.com/download")
        print("   - Or run: brew install ollama")
        sys.exit(1)
    
    print("✅ Ollama is installed")
    
    # Check if server is already running
    if is_ollama_running():
        print("✅ Ollama server is already running")
    else:
        # Try to start the server
        if not start_ollama_server():
            print("\n🔧 Manual setup required:")
            print("1. Start Ollama server in a separate terminal:")
            print("   ollama serve")
            print("2. Then run this script again")
            sys.exit(1)
    
    # Download the model
    print()
    if download_qwen_model():
        print("\n🎉 Setup complete! You can now run Chaplin with text correction.")
        print("Run: python main_safe.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe")
    else:
        print("\n⚠️  Model download failed, but you can still run Chaplin without text correction.")
        print("Run: python main_safe.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe")

if __name__ == "__main__":
    main()