import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Home.css';

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="home-container">
      {/* Silver-Blue Gradient Animated Heading with Clean Word Spacing */}
      <h1 className="lipbuddy-gradient">
        <span className="word">Welcome</span>
        <span className="word">to</span>
        <span className="word">Lip</span>
        <span className="word">Buddy</span>
      </h1>

      {/* Subtitle */}
      <p className="home-subtitle">
        Upload videos. Read lips. Decode silent speech with AI.
      </p>

      {/* Buttons */}
      <div className="home-buttons">
        <button className="home-button" onClick={() => navigate("/login")}>
          Get Started
        </button>
        <button className="home-button outline" onClick={() => navigate("/about")}>
          Learn More
        </button>
      </div>
    </div>
  );
};

export default Home;