# 🧠 Lip Buddy Project

## 1. Lip Buddy Frontend
```bash
cd lipbuddy/frontend/src
npm start
```

## 2. LipNet (Streamlit App)
```bash
cd LipNet-main/app
streamlit run streamlitapp.py
```

## 3. LipNet (Flask App)
```bash
cd LipNet-main
python app.py
```

## 4. Lip Main (Advanced AI Lip Reading)
**Recommended Web Interface:**
```bash
cd lipmain
python app_webcam.py config_filename=./configs/LRS3_V_WER19.1.ini detector=retinaface
```
*Open http://localhost:5005 in your browser.*

**CLI Version:**
```bash
cd lipmain
python main_safe.py config_filename=./configs/LRS3_V_WER19.1.ini detector=retinaface
```

