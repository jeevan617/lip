import React from 'react';
import { useNavigate } from 'react-router-dom';
import './History.css'; // Import styles

const History = ({ history }) => {
  const navigate = useNavigate();

  return (
    <div className="history-container">
      <button className="back-button" onClick={() => navigate(-1)}>
        ← Back
      </button>

      <h3>Purpose</h3>

      {history.length === 0 ? (
        <p>infomation about lipbuddy</p>
      ) : (
        <ul>
          {history.map((prediction, index) => (
            <li key={index} className="history-item">
              <img
                src={prediction.image}
                alt="Uploaded Preview"
                className="history-image"
              />
              <div className="history-details">
                <p>{prediction.result}</p>
                <p>Confidence: {prediction.confidence.toFixed(2)}%</p>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default History;