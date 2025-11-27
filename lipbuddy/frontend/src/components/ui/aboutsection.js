// src/components/AboutSection.js
import React from 'react';
import './aboutsection.css';// Optional styling file

const aboutItems = [
  {
    img: '/images/mission.jpg',
    title: 'Our Mission',
    description: 'We aim to simplify workflows through intuitive design and powerful tools.',
  },
  {
    img: '/images/team.jpg',
    title: 'Our Team',
    description: 'A passionate group of developers, designers, and thinkers.',
  },
  {
    img: '/images/future.jpg',
    title: 'Looking Ahead',
    description: 'We are constantly evolving to meet tomorrow’s challenges.',
  },
];

function AboutSection() {
  return (
    <div className="about-section">
      {aboutItems.map((item, index) => (
        <div key={index} className="about-item">
          <img src={item.img} alt={item.title} />
          <h3>{item.title}</h3>
          <p>{item.description}</p>
        </div>
      ))}
    </div>
  );
}

export default AboutSection;