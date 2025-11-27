import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './login.css'; // Make sure this file is updated as we discussed

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('upload'); // 'upload' or 'live'
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const res = await fetch("http://localhost:5050/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      const data = await res.json();

      setTimeout(() => {
        setLoading(false);

        if (res.ok) {
          const target = mode === 'upload' ? 'http://localhost:8501' : 'http://localhost:5005';
          window.location.href = target; // Redirect based on selected mode
        } else {
          setError(data.error || "Login failed");
        }
      }, 1000); // 1-second delay for loading animation
    } catch (err) {
      setLoading(false);
      setError("Server error");
    }
  };

  return (
    <>
      {/* Fullscreen loading animation */}
      {loading && (
        <div className="loading-screen">
          <div className="loading-text">
            {'Logging in to your Lip Buddy account...'.split('').map((char, i) => (
              <span key={i} style={{ animationDelay: `${i * 0.05}s` }}>{char}</span>
            ))}
          </div>
        </div>
      )}

      {/* Login Form */}
      <div className="auth-container">
        <h2 className="login-title">WELCOME</h2>

        <input
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          disabled={loading}
        />

        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          disabled={loading}
        />

        {/* Mode selector */}
        <div className="mode-selector">
          <button
            type="button"
            className={mode === 'upload' ? 'selected' : ''}
            onClick={() => setMode('upload')}
            disabled={loading}
          >
            Upload Prediction
          </button>
          <button
            type="button"
            className={mode === 'live' ? 'selected' : ''}
            onClick={() => setMode('live')}
            disabled={loading}
          >
            Live Prediction
          </button>
        </div>

        {!loading && error && <p className="error">{error}</p>}

        <button className="fancy-button" onClick={handleLogin} disabled={loading}>
          {loading ? 'Please wait...' : 'Login'}
        </button>

        <button className="link-button" onClick={() => navigate('/signup')} disabled={loading}>
          Don't have an account? <strong>Sign Up</strong>
        </button>
      </div>
    </>
  );
};

export default Login;