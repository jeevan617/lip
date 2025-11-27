import tensorflow as tf
from typing import List
import cv2
import os

# Define the vocabulary for character mapping
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Create lookup tables
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# Load and preprocess the video (extract mouth region and normalize)
def load_video(path: str) -> tf.Tensor:
    cap = cv2.VideoCapture(path)
    frames = []

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        frame = tf.image.rgb_to_grayscale(frame)

        # Crop to mouth region (190:236 height, 80:220 width)
        cropped = frame[190:236, 80:220, :]
        frames.append(cropped)

    cap.release()

    # Stack into tensor and normalize
    frames = tf.stack(frames)
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))

    return tf.cast((frames - mean), tf.float32) / std

# Load and preprocess alignment (transcription labels)
def load_alignments(path: str) -> tf.Tensor:
    with open(path, 'r') as f:
        lines = f.readlines()

    tokens = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[2] != 'sil':
            tokens.append(' ')         # Insert space between words
            tokens.append(parts[2])    # Append the spoken token

    # Convert list of strings to tensor of character indices
    joined = tf.strings.unicode_split(tokens, input_encoding='UTF-8')
    return char_to_num(tf.reshape(joined, (-1)))[1:]  # Skip blank token

# Unified loader for both video and alignment
def load_data(path: tf.Tensor):
    path = bytes.decode(path.numpy())
    file_name = os.path.splitext(os.path.basename(path))[0]

    # Paths for video and alignment
    base_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    video_path = os.path.join(base_data_path, 's1', f'{file_name}.mpg')
    align_path = os.path.join(base_data_path, 's2', f'{file_name}.align')

    print("Video Path:", video_path)
    print("Alignment Path:", align_path)

    # Ensure files exist
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(align_path):
        raise FileNotFoundError(f"Alignment file not found: {align_path}")

    video = load_video(video_path)
    alignment = load_alignments(align_path)

    return video, alignment