# Lip Buddy - AI Lip Reading Assistant

Lip Buddy is an advanced AI-powered application that performs lip reading from video input. It uses visual speech recognition (VSR) to transcribe speech from lip movements and enhances the output using a Large Language Model (Ollama) for correction.

## 🚀 Features

-   **Real-time Video Recording**: Capture video directly from your webcam via a modern web interface.
-   **Visual Speech Recognition**: State-of-the-art lip reading model (based on Auto-AVSR).
-   **AI Text Correction**: Uses Ollama (Qwen model) to correct grammar and context of the raw transcription.
-   **Robust Face Detection**: Uses RetinaFace for accurate 5-point facial landmark detection.
-   **Modern UI**: A beautiful, dark-themed interface with glassmorphism effects and animations.

## 🛠️ Prerequisites

-   **Python 3.8+**
-   **Ollama**: For text correction (optional but recommended).
    -   Download from [ollama.com](https://ollama.com)
    -   Pull the model: `ollama pull qwen3:4b` (or any other model you prefer, update code if needed)
-   **FFmpeg**: Required for video processing.

## 📦 Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <your-repo-url>
    cd lipmain
    ```

2.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have PyTorch installed with CUDA support if you have a GPU, otherwise CPU version will work but slower.)*

3.  **Download Weights**:
    -   Ensure `mobilenet0.25_Final.pth` is in `Pytorch_Retinaface/weights/`.
    -   Ensure the VSR model weights are in the correct path as defined in your config (e.g., `LRS3_V_WER19.1.ini`).

## 🏃‍♂️ Usage

### Web Interface (Recommended)

1.  **Start the Application**:
    ```bash
    python app_webcam.py config_filename=./configs/LRS3_V_WER19.1.ini detector=retinaface
    ```

2.  **Open in Browser**:
    Go to **[http://localhost:5005](http://localhost:5005)**

3.  **Use the App**:
    -   Click **"Start Recording"**.
    -   Speak clearly into the camera for at least 2 seconds.
    -   Click **"Stop"**.
    -   Wait for the processing to complete. The raw and corrected text will appear on the screen.

### Command Line Interface

If you prefer the terminal:
```bash
python main_safe.py config_filename=./configs/LRS3_V_WER19.1.ini detector=retinaface
```
-   Press and hold **ALT/OPTION** to record.
-   Press **'q'** to quit.

## 🔧 Configuration

-   **Configs**: Check the `configs/` directory for model configurations.
-   **Hydra**: The app uses Hydra for configuration management. You can override parameters via command line.

## ⚠️ Troubleshooting

-   **"Ollama not available"**: Ensure Ollama is running (`ollama serve`). The app will still work but without text correction.
-   **"Landmarks mismatch"**: If face detection fails, the app will try a fallback method. Ensure good lighting and face the camera directly.
-   **Port already in use**: If port 5005 is busy, kill the process using it or change the port in `app_webcam.py`.

## 📝 License

[Jeevan M]
]
