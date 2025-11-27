import streamlit as st
import os
import imageio
import tensorflow as tf
import random
import time

from utils import load_data
from modelutil import load_model
from model import load_model

st.set_page_config(layout='wide')

# Sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')
    weight_option = st.selectbox("Select model weights", ["overlapped", "unseen"])

# Title
st.title('LipNet Demo (AI Lip Reading Simulation)')

# Set data paths
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 's1'))
align_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 's2'))

if not os.path.exists(data_dir):
    st.error("Video folder not found.")
    st.stop()

# Filter valid videos
video_files = [
    f for f in os.listdir(data_dir)
    if f.endswith('.mpg') and os.path.exists(os.path.join(align_dir, f.replace('.mpg', '.align')))
]
selected_video = st.selectbox("Choose a video file", video_files)

col1, col2 = st.columns([1, 1])

if st.button("Run Lip Reading"):
    file_path = os.path.join(data_dir, selected_video)

    # --- LEFT: Video ---
    with col1:
        st.markdown("### 🎥 Original Video")
        os.system(f'ffmpeg -i "{file_path}" -vcodec libx264 test_video.mp4 -y')
        with open("test_video.mp4", "rb") as f:
            st.video(f.read())

    # --- RIGHT: Binary + Processed Frames ---
    with col2:
        st.markdown("### 🧠 Neural Activity (Model Input)")

        video, _ = load_data(tf.convert_to_tensor(file_path))
        video_np = video.numpy()
        frames = [tf.squeeze(frame * 255).numpy().astype('uint8') for frame in video_np]
        imageio.mimsave('animation.gif', frames, fps=10)
        st.image('animation.gif', width=400)

        # Binary simulation
        def simulate_binary_activity(frames=40, width=60, height=12, delay=0.15):
            placeholder = st.empty()
            for _ in range(frames):
                binary_block = '\n'.join(
                    ''.join(random.choice(['0', '1', ' ']) for _ in range(width))
                    for _ in range(height)
                )
                placeholder.code(binary_block, language='bash')
                time.sleep(delay)
            placeholder.empty()

        simulate_binary_activity()

    # Align output
    video_id = os.path.splitext(selected_video)[0]
    align_path = os.path.join(align_dir, f"{video_id}.align")

    st.subheader("🔡 Model Output (Raw Tokens):")
    if os.path.exists(align_path):
        with open(align_path, 'r') as f:
            raw_lines = f.readlines()
            raw_tokens = [line.strip().split()[2] for line in raw_lines if len(line.strip().split()) >= 3]
            st.code(raw_tokens)

        # Decoded words (typewriter effect)
        decoded_text = ' '.join(raw_tokens).replace('sil', '').replace('sp', '').strip()
        st.subheader("🧾 Decoded Prediction:")
        placeholder = st.empty()
        typed = ""

        for word in decoded_text.split():
            typed += word + " "
            placeholder.success(typed.strip())
            time.sleep(0.6)  # ⏳ slightly slower
    else:
        st.warning("No `.align` file found for this video.")