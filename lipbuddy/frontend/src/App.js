import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Home from './pages/Home';
import About from './pages/About';
import Devs from './pages/Devs';
import MainApp from './MainApp';
import ResultPage from './pages/ResultPage';
import ChatButton from './pages/ChatButton';
import Chatbot from './pages/Chatbot';
import History from './pages/History'; // Import History page
import './App.css';

function App() {
  const [showChatbot, setShowChatbot] = useState(false);
  const [history, setHistory] = useState([]); // History state

  // Toggle chat visibility
  const toggleChat = () => {
    setShowChatbot(prev => !prev);
  };

  return (
    <Router>
      <div>
        {/* Navbar */}
        <nav className="navbar">
          <Link to="/">Home</Link>
          <Link to="/about">About</Link>
          <Link to="/login">Login</Link>
          <Link to="/signup">Signup</Link>
          <Link to="/devs">Devs</Link>
          <Link to="/history">Purpose</Link> {/* History Button */}
        </nav>

        {/* Floating Chat Button */}
        <ChatButton onClick={toggleChat} />

        {/* Chatbot Popup */}
        {showChatbot && <Chatbot onClose={toggleChat} />}

        {/* Routes */}
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/devs" element={<Devs />} />
          <Route path="/main-app" element={<MainApp setHistory={setHistory} />} /> {/* Pass setHistory */}
          <Route path="/result" element={<ResultPage />} />
          <Route path="/history" element={<History history={history} />} /> {/* History route */}
        </Routes>
      </div>
    </Router>
  );
}

export default App;