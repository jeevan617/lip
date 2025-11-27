import React, { useState, useEffect } from "react";
import './ChatButton.css';

const ChatButton = () => {
  const [showChat, setShowChat] = useState(false);
  const [answerText, setAnswerText] = useState(null);

  const questions = [
    { id: 1, text: "What is Lip Reading AI?", answer: "Lip Reading AI is a system that interprets speech by analyzing lip movements using computer vision." },
    { id: 2, text: "How does Lip Reading work?", answer: "It uses deep learning models like CNNs and RNNs to map lip movements to words." },
    { id: 3, text: "What datasets are used?", answer: "Popular datasets include GRID, LRW (Lip Reading in the Wild), and LRS3." },
    { id: 4, text: "Is it better than audio speech recognition?", answer: "Lip reading AI can assist audio models, especially in noisy environments, but it's not a complete replacement." },
    { id: 5, text: "Can lip reading be used in security?", answer: "Yes, it can help in surveillance, silent speech interfaces, and accessibility tools." },
    { id: 6, text: "What technologies are used?", answer: "It uses computer vision, deep learning (CNN, RNN, Transformer), and video processing." },
    { id: 7, text: "What is the accuracy of lip reading AI?", answer: "Accuracy depends on dataset and model, but state-of-the-art models can reach over 80% on controlled data." },
    { id: 8, text: "Can it work in real-time?", answer: "Yes, with optimization, lip reading models can process video frames in real-time." },
    { id: 9, text: "Is lip reading AI used in any products?", answer: "It's being explored in accessibility apps, smart assistants, and video conferencing tools." },
    { id: 10, text: "How can I build a lip reading model?", answer: "Start by collecting video data, preprocessing frames, and training a model using deep learning frameworks like PyTorch or TensorFlow." }
  ];

  const handleQuestionClick = (answer) => {
    setAnswerText(answer);
  };

  const handleChatOpen = () => {
    setShowChat(true);
  };

  useEffect(() => {
    if (showChat) {
      const timeout = setTimeout(() => {
        // Animation or other effects can be placed here
      }, 1000);

      return () => clearTimeout(timeout);
    }
  }, [showChat]);

  return (
    <>
      <button className="chat-btn" onClick={handleChatOpen}>
        💬
      </button>

      {showChat && (
        <div className="chat-container fade-in">
          <div className="chat-box">
            <div className="chat-header">
              <h3>Lip Reading Chatbot</h3>
              <button className="close-chat" onClick={() => setShowChat(false)}>X</button>
            </div>

            <div className="chat-content">
              {answerText ? (
                <div className="chat-answer">
                  <p>{answerText}</p>
                  <button onClick={() => setAnswerText(null)}>Back</button>
                </div>
              ) : (
                questions.map((question) => (
                  <div
                    key={question.id}
                    className="chat-bubble"
                    onClick={() => handleQuestionClick(question.answer)}
                  >
                    {question.text}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatButton;