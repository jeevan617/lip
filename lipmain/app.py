from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import torch
import hydra
from omegaconf import OmegaConf
from pipelines.pipeline import InferencePipeline
from chaplin import Chaplin
import os
import base64
import numpy as np
import cv2
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10*1024*1024)  # 10MB max

# Global variable to hold the chaplin instance
chaplin = None

def load_model(cfg):
    global chaplin
    chaplin = Chaplin()
    # Disable face tracking - face detection fails even with 0.1 threshold
    # User must center their face in webcam for center crop to work
    chaplin.vsr_model = InferencePipeline(
        cfg.config_filename, device=torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available(
        ) and cfg.gpu_idx >= 0 else "cpu"), detector=cfg.detector, face_track=False)
    print("\n\033[48;5;22m\033[97m\033[1m MODEL LOADED SUCCESSFULLY! \033[0m\n")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('test_connection')
def test_connection(data):
    print(f'✅ Test connection received: {data}')
    emit('test_response', {'status': 'Socket.IO is working!'})

@socketio.on('video_frame')
def handle_video_frame(data):
    print(f"Received video_frame event, data length: {len(data) if data else 0}")
    
    if chaplin is None:
        print("Error: Chaplin model not initialized")
        emit('processed_text', {'raw': 'Error: Model not loaded', 'corrected': 'Error: Model not loaded'})
        return

    try:
        # Decode the video frame
        video_data = base64.b64decode(data.split(',')[1])
        
        # Save the video data to a temporary file
        temp_video_path = f"webcam{time.time_ns() // 1_000_000}.webm"
        with open(temp_video_path, 'wb') as f:
            f.write(video_data)
        
        file_size = os.path.getsize(temp_video_path)
        print(f"Saved video to {temp_video_path}, size: {file_size} bytes")
        
        # Convert WebM to grayscale MP4 at 16 FPS (like standalone)
        converted_path = temp_video_path.replace('.webm', '_converted.mp4')
        print(f"Converting to grayscale MP4 at 16 FPS...")
        
        cap = cv2.VideoCapture(temp_video_path)
        fps = 16  # Match standalone FPS
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(
            converted_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height),
            False  # Grayscale
        )
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out.write(gray)
            frame_count += 1
        
        cap.release()
        out.release()
        print(f"Converted {frame_count} frames to grayscale MP4")
        
        # Clean up original WebM
        os.remove(temp_video_path)
        temp_video_path = converted_path
        
        print(f"Processing video: {temp_video_path}")
        
        # Perform inference
        print("Running VSR model...")
        result = chaplin.perform_inference(temp_video_path)
        print(f"Inference complete. Raw output: {result['output']}")
        
        # Clean up the temporary file
        os.remove(temp_video_path)
        print(f"Cleaned up {temp_video_path}")

        # Emit the raw result to the frontend immediately
        emit('processed_text', {'raw': result['output'], 'corrected': 'Correcting...'})

        # Get the corrected text (blocking)
        print("Waiting for LLM correction...")
        try:
            corrected_text = result["corrected_text_future"].result()
            print(f"LLM correction complete: {corrected_text}")
        except ConnectionError as e:
            # Ollama not running - use raw output as corrected text
            print(f"⚠️  Ollama not available: {e}")
            print("💡 Using raw output without AI correction")
            corrected_text = result['output']

        # Emit the corrected result
        emit('processed_text', {'raw': result['output'], 'corrected': corrected_text})
        
    except IndexError as e:
        print(f"Video too short error: {str(e)}")
        emit('processed_text', {
            'raw': 'Error: Video too short', 
            'corrected': 'Please record for at least 2-3 seconds while speaking clearly.'
        })
        # Clean up the file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        emit('processed_text', {'raw': f'Error: {str(e)}', 'corrected': 'Processing failed'})
        # Clean up the file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)



if __name__ == '__main__':
    hydra.initialize(config_path="hydra_configs", version_base=None)
    cfg = hydra.compose(config_name="default")
    load_model(cfg)
    print("Starting socketio on port 5005...")
    socketio.run(app, port=5005, host='0.0.0.0')