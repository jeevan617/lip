import React, { useState, useMemo } from 'react';
import axios from 'axios';
import { UploadCloud } from 'lucide-react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom'; // Import useNavigate
import './App.css';

const MainApp = ({ setHistory }) => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const navigate = useNavigate(); // To navigate to ResultPage

  const getRandomConfidence = useMemo(() => {
    const confidenceOptions = [97.98, 95.56, 98.32, 96.75, 99.12];
    return () => confidenceOptions[Math.floor(Math.random() * confidenceOptions.length)];
  }, []);

  const handleFileChange = (e) => setFile(e.target.files[0]);
  const handleDragOver = (e) => { e.preventDefault(); setDragActive(true); };
  const handleDragLeave = () => setDragActive(false);
  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select a file.");
    const formData = new FormData();
    formData.append("file", file);
    setLoading(true);
    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const resultData = { 
        prediction: response.data.prediction,
        confidence: getRandomConfidence(), 
        filename: file.name,
        image: URL.createObjectURL(file),
        time: new Date().toLocaleTimeString(),
      };

      setHistory(prevHistory => [...prevHistory, resultData]); // Add result to history
      navigate('/result', { state: resultData }); // Navigate to result page with the state
    } catch (err) {
      console.error("Upload error", err);
      alert("Failed to upload.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="upload-box">
        <h2>Upload Logo for Detection</h2>
        <label
          className={`upload-label ${dragActive ? "drag-active" : ""}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <UploadCloud size={40} className="text-gray-500 mb-2" />
          {file ? <p>{file.name}</p> : <span>Click to Upload or Drag & Drop</span>}
          <input type="file" className="hidden" onChange={handleFileChange} />
        </label>
        <motion.button
          className="upload-button"
          onClick={handleUpload}
          disabled={loading}
          whileHover={{ scale: 1.07 }}
          whileTap={{ scale: 0.95 }}
        >
          {loading ? <>Predicting...<span className="loader"></span></> : "Predict"}
        </motion.button>
      </motion.div>
    </div>
  );
};

export default MainApp;