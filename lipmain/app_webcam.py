#!/usr/bin/env python3
"""
Web version using standalone's webcam recording approach
"""
import torch
import hydra
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from chaplin_safe import Chaplin
import cv2
import time
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10*1024*1024)

chaplin = None
recording = False
video_writer = None
output_path = None
frame_count = 0

@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def load_model(cfg):
    global chaplin
    
    # Check for Ollama availability
    use_ollama = True
    try:
        from ollama import Client
        client = Client()
        client.list()
        print("✅ Ollama server detected - text correction enabled")
    except Exception as e:
        print(f"⚠️  Ollama not available: {e}")
        print("💡 Running without text correction - raw output will be typed")
        use_ollama = False

    # Initialize Chaplin with safe settings
    chaplin = Chaplin(use_ollama=use_ollama)
    
    # Load VSR model
    print("\n🔄 Loading VSR model...")
    device = torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() and cfg.gpu_idx >= 0 else "cpu")
    print(f"📱 Using device: {device}")
    
    from pipelines.pipeline import InferencePipeline
    chaplin.vsr_model = InferencePipeline(
        cfg.config_filename,
        device=device,
        detector=cfg.detector,
        face_track=True
    )
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

@socketio.on('start_recording')
def start_recording():
    global recording, video_writer, output_path, frame_count
    recording = True
    frame_count = 0
    output_path = f"webcam{time.time_ns() // 1_000_000}.mp4"
    video_writer = None  # Initialize lazily
    
    print(f"Started recording session: {output_path}")
    emit('recording_started', {'status': 'Recording started'})

@socketio.on('video_frame')
def handle_frame(data):
    global recording, video_writer, frame_count, output_path
    
    if not recording:
        return
    
    # Decode base64 frame
    import base64
    import numpy as np
    
    try:
        # Handle data URI scheme if present
        if ',' in data:
            data = data.split(',')[1]
            
        frame_data = base64.b64decode(data)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("Failed to decode frame")
            return
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize video writer if needed
        if video_writer is None:
            height, width = gray.shape
            print(f"Initializing video writer with {width}x{height}")
            video_writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                25,  # Use 25 FPS as set in frontend
                (width, height),
                False  # Grayscale
            )
        
        # Write frame
        video_writer.write(gray)
        frame_count += 1
        
    except Exception as e:
        print(f"Error handling frame: {e}")

@socketio.on('stop_recording')
def stop_recording():
    global recording, video_writer, output_path, frame_count
    
    if not recording:
        return
    
    recording = False
    if video_writer:
        video_writer.release()
    
    print(f"Stopped recording: {output_path}, {frame_count} frames")
    
    # Check minimum duration (2 seconds at 16 FPS = 32 frames)
    if frame_count < 32:
        print(f"Video too short ({frame_count} frames), need at least 32")
        os.remove(output_path)
        emit('processed_text', {
            'raw': f'Video too short ({frame_count} frames)',
            'corrected': 'Please record for at least 2 seconds'
        })
        return
    
    # Process video using standalone's method
    try:
        print(f"Processing {frame_count} frames...")
        result = chaplin.perform_inference(output_path)
        
        raw_text = result['output']
        print(f"\n RAW OUTPUT : {raw_text}\n")
        
        # Clean up video file
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # Perform correction if Ollama is enabled
        corrected_text = raw_text
        if chaplin.use_ollama:
            print("🤖 Correcting text with Ollama...")
            corrected_text = correct_text(raw_text)
            print(f"✨ Corrected: {corrected_text}")
        
        # Send result
        emit('processed_text', {
            'raw': raw_text,
            'corrected': corrected_text
        })
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        emit('processed_text', {
            'raw': f'Error: {str(e)}',
            'corrected': 'Processing failed'
        })

def correct_text(text):
    """Correct text using Ollama"""
    try:
        from ollama import Client
        client = Client()
        
        prompt = f"""You are an assistant that helps make corrections to the output of a lipreading model. 
The text you will receive was transcribed using a video-to-text system that attempts to lipread the subject speaking in the video, so the text will likely be imperfect. 
The input text will also be in all-caps, although your respose should be capitalized correctly and should NOT be in all-caps.

If something seems unusual, assume it was mistranscribed. Do your best to infer the words actually spoken, and make changes to the mistranscriptions in your response. 
Do not add more words or content, just change the ones that seem to be out of place.
Also, add correct punctuation to the entire text. ALWAYS end each sentence with the appropriate sentence ending: '.', '?', or '!'.

Return ONLY the corrected text. Do not include any explanations or JSON.

Transcription:
{text}
"""
        response = client.chat(model='qwen3:4b', messages=[
            {'role': 'user', 'content': prompt}
        ])
        
        return response['message']['content'].strip()
    except Exception as e:
        print(f"⚠️  Correction failed: {e}")
        return text

if __name__ == '__main__':
    # Load model
    load_model()
    
    # Start server
    print("Starting socketio on port 5005...")
    socketio.run(app, port=5005, host='0.0.0.0')
