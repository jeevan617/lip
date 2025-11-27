import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import './ResultPage.css'; // Ensure this file exists

const ResultPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { prediction, confidence, filename, image, time } = location.state || {};

  // Debug: Print received data
  console.log("Received state:", location.state);

  // Navigate to MainApp for new prediction
  const handleRefresh = () => {
    navigate("/main-app");
  };

  return (
    <div className="result-container">
      <h1 className="result-header">Prediction Result</h1>

      <div className="result-details">
        <div className="result-item">
          <strong>Prediction: </strong>
          <span>{prediction ? (prediction === "Real Logo" ? "✅ Real Logo" : "❌ Fake Logo") : "No prediction"}</span>
        </div>

        <div className="result-item">
          <strong>Confidence: </strong>
          <span>{confidence !== undefined ? confidence.toFixed(2) : "No confidence"}</span>
        </div>

        <div className="result-item">
          <strong>Uploaded File: </strong>
          <span>{filename || "No file uploaded"}</span>
        </div>

        <div className="result-item">
          <strong>Prediction Time: </strong>
          <span>{time || "Not available"}</span>
        </div>
      </div>

      {image && (
        <div className="result-preview">
          <h3>Image Preview</h3>
          <img src={image} alt="Uploaded Logo" className="result-image" />
        </div>
      )}

      {/* 🔄 Refresh Button */}
      <div className="refresh-button-container">
        <button className="refresh-button" onClick={handleRefresh}>
          🔄 Refresh Prediction
        </button>
      </div>
    </div>
  );
};

export default ResultPage;