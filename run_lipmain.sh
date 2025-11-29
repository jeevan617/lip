#!/bin/bash
echo "Starting Lipmain on port 5001..."
cd lipmain
python app_webcam.py config_filename=./configs/LRS3_V_WER19.1.ini detector=retinaface
