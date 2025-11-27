import React from "react";
import "./About.css";

import img1 from "../assets/img1.jpeg"; // Replace with actual image filenames
import img2 from "../assets/img2.jpeg";
import img3 from "../assets/img3.jpeg";

function About() {
  return (
    <div className="about-container">
      <h1 className="big-title">
        {[
          "A", "b", "o", "u", "t", " ", "L", "i", "p", " ", "B", "u", "d", "d", "y"
        ].map((char, i) => (
          <span
            className="gradient-letter"
            key={i}
            style={{ animationDelay: `${i * 0.05}s` }}
          >
            {char === " " ? "\u00A0" : char}
          </span>
        ))}
      </h1>

      <div className="about-section left">
        <div className="about-text-wrapper">
          <p className="about-paragraph animated-text">
            Lip Buddy is a cutting-edge AI tool designed to understand speech by analyzing lip movements. Using computer vision and deep learning, it can interpret what someone is saying even in the absence of audio.
          </p>
        </div>
        <div className="about-image-wrapper">
          <div className="image-box">
            <img className="about-image" src={img1} alt="Lip Reading AI" />
          </div>
        </div>
      </div>

      <div className="about-section right">
        <div className="about-image-wrapper">
          <div className="image-box">
            <img className="about-image" src={img2} alt="AI Lip Recognition" />
          </div>
        </div>
        <div className="about-text-wrapper">
          <p className="about-paragraph animated-text">
            Trained on large-scale datasets, Lip Buddy uses neural networks to detect, track, and translate lip movements into accurate text. It’s especially useful in noisy environments, surveillance, and accessibility tech.
          </p>
        </div>
      </div>

      <div className="about-section left">
        <div className="about-text-wrapper">
          <p className="about-paragraph animated-text">
            Whether you're building assistive apps, silent speech interfaces, or exploring lip synchronization for avatars and animations, Lip Buddy empowers you with reliable and real-time lip reading capabilities.
          </p>
        </div>
        <div className="about-image-wrapper">
          <div className="image-box">
            <img className="about-image" src={img3} alt="Lip Sync Detection" />
          </div>
        </div>
      </div>
    </div>
  );
}

export default About;