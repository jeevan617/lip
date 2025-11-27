import React, { useState } from 'react';
import './Chatbot.css';

const questionsData = [
  { q: 'What is Lip Reading AI?', a: 'Lip Reading AI is the process of understanding speech by analyzing lip movements using computer vision and deep learning.' },
  { q: 'How does it work?', a: 'It uses video frames to detect lip movements, which are then processed by neural networks to predict the spoken words.' },
  { q: 'Who can use it?', a: 'It’s useful for accessibility apps, surveillance systems, and noisy environments where audio speech is hard to detect.' },
  { q: 'Is it 100% accurate?', a: 'Lip reading AI is improving but not perfect. Accuracy depends on lighting, resolution, and model training.' },
  { q: 'Do I need special hardware?', a: 'No, it can work with regular cameras, but high-resolution video improves accuracy.' },
  { q: 'What datasets are used?', a: 'Popular datasets include GRID, LRW (Lip Reading in the Wild), and LRS3 for training models.' },
  { q: 'Is it free?', a: 'Some models and research code are open-source. Applications may vary depending on the provider.' },
  { q: 'Can I upload my own video?', a: 'Yes, in most demos or systems, you can upload a video clip for lip reading analysis.' },
  { q: 'Is real-time lip reading possible?', a: 'Yes, with optimized models and good hardware, real-time processing is achievable.' },
  { q: 'Other', a: 'For any other queries, please contact us at lipreadingai@example.com.' },
];

const Chatbot = () => {
  const [selectedIndex, setSelectedIndex] = useState(null);

  const handleBack = () => {
    setSelectedIndex(null);
  };

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">👄 Lip Reading Assistant</div>
      <div className="chatbot-content">
        {selectedIndex === null ? (
          <div className="bubbles">
            {questionsData.map((item, index) => (
              <div
                key={index}
                className="bubble"
                onClick={() => setSelectedIndex(index)}
              >
                {item.q}
              </div>
            ))}
          </div>
        ) : (
          <div className="answer-view">
            <div className="answer-text">{questionsData[selectedIndex].a}</div>
            <button className="back-button" onClick={handleBack}>
              ⬅ Back
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Chatbot;