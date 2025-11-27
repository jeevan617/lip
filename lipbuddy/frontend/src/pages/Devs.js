import React from 'react';
import './Devs.css';

import sanjanaImage from '../assets/Sanjana.jpg';
import meghanaImage from '../assets/Meghana.jpg';
import darshanImage from '../assets/Darshan.jpg';
import shwethaImage from '../assets/Shwetha.jpg'; // Replace if there's a separate image for Shwetha

const Devs = () => {
  const team = [
    {
      name: 'L. Sanjana',
      usn: '4MN22CS021',
      lang: 'C, C++, Python, HTML, CSS, Java',
      img: sanjanaImage,
    },
    {
      name: 'KN. Meghana',
      usn: '4MN22CS020',
      lang: 'C, Basic Python, HTML, CSS, Java',
      img: meghanaImage,
    },
    {
      name: 'Shwetha M',
      usn: '4MN23CS404',
      lang: 'Python, HTML, CSS, Java, C',
      img: shwethaImage,
    },
    {
      name: 'Darshan D.M',
      usn: '4MN22CS014',
      lang: 'C, C++, Python, HTML, CSS, Java',
      img: darshanImage,
    },
  ];

  return (
    <div className="devs-container">
      <h1>👩‍💻 Meet the Developers</h1>
      <p>We are the team behind Lip Buddy – your silent speech AI assistant!</p>

      <div className="team-grid">
        {team.map((dev, index) => (
          <div className="dev-card" key={index}>
            <img src={dev.img} alt={dev.name} className="dev-img" />
            <h3>{dev.name}</h3>
            <p><strong>USN:</strong> {dev.usn}</p>
            <p><strong>Languages:</strong> {dev.lang}</p>
          </div>
        ))}
      </div>

      <h2>📬 Contact Information</h2>
      <p>Email: <a href="mailto:lipbuddy.support@gmail.com">lipbuddy.support@gmail.com</a></p>
      <p>Website: <a href="https://lipbuddy.vercel.app" target="_blank" rel="noopener noreferrer">lipbuddy.vercel.app</a></p>

      <p className="footer-msg">Feel free to reach out if you have any questions or feedback!</p>
    </div>
  );
};

export default Devs;