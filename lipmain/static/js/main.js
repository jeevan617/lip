const video = document.getElementById('video');
const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
const rawOutput = document.getElementById('raw-output');
const correctedOutput = document.getElementById('corrected-output');
const statusText = document.getElementById('status-text');
const connectionDot = document.getElementById('connection-dot');
const processingOverlay = document.getElementById('processing-overlay');

const socket = io();

let stream;
let recordingInterval;
const FPS = 25;

// Helper to update status
function updateStatus(message, type = 'neutral') {
    statusText.textContent = message;
    connectionDot.className = 'status-dot'; // reset
    if (type === 'connected') connectionDot.classList.add('connected');
    if (type === 'recording') connectionDot.classList.add('recording');
}

// Initialize camera
async function initCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: FPS }
            },
            audio: false
        });
        video.srcObject = stream;
        console.log('📷 Camera initialized');
        updateStatus('Ready to record', 'connected');
    } catch (error) {
        console.error('❌ Error accessing camera:', error);
        updateStatus('Error: Camera access denied', 'error');
        alert('Could not access camera. Please allow camera permissions.');
    }
}

// Canvas for frame capture
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

startButton.addEventListener('click', () => {
    if (!stream) {
        initCamera();
        return;
    }

    console.log('🎬 Start recording');
    socket.emit('start_recording');

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Start frame streaming
    recordingInterval = setInterval(() => {
        ctx.drawImage(video, 0, 0);
        const dataURL = canvas.toDataURL('image/jpeg', 0.7);
        socket.emit('video_frame', dataURL);
    }, 1000 / FPS);

    updateStatus('Recording... (speak for at least 2 seconds)', 'recording');
    document.querySelector('.video-wrapper').classList.add('recording');

    startButton.disabled = true;
    stopButton.disabled = false;

    // Clear outputs
    rawOutput.innerHTML = '<span class="placeholder">Recording...</span>';
    correctedOutput.innerHTML = '<span class="placeholder">Recording...</span>';

    processingOverlay.classList.remove('active');
});

stopButton.addEventListener('click', () => {
    console.log('⏹️ Stop recording');
    clearInterval(recordingInterval);
    socket.emit('stop_recording');

    updateStatus('Processing video...', 'connected');
    document.querySelector('.video-wrapper').classList.remove('recording');
    processingOverlay.classList.add('active');

    startButton.disabled = false;
    stopButton.disabled = true;

    // Show processing indicators
    rawOutput.innerHTML = '<span class="placeholder"><i class="fas fa-spinner fa-spin"></i> Analyzing...</span>';
    correctedOutput.innerHTML = '<span class="placeholder"><i class="fas fa-spinner fa-spin"></i> Correcting...</span>';
});

socket.on('processed_text', (data) => {
    console.log('📝 Received result:', data);

    if (data.raw) {
        rawOutput.textContent = data.raw;
        rawOutput.classList.remove('placeholder');
    }

    if (data.corrected) {
        correctedOutput.textContent = data.corrected;
        correctedOutput.classList.remove('placeholder');
    }

    processingOverlay.classList.remove('active');
    updateStatus('Ready to record', 'connected');
});

socket.on('connect', () => {
    console.log('✅ Connected to server');
    updateStatus('Connected', 'connected');
    initCamera();
});

socket.on('disconnect', () => {
    console.log('❌ Disconnected from server');
    updateStatus('Disconnected', 'neutral');
    clearInterval(recordingInterval);
});

socket.on('recording_started', () => {
    console.log('Server confirmed recording started');
});