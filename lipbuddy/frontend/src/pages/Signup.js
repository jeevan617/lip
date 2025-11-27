import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './signup.css';

const Signup = () => {
  const [formData, setFormData] = useState({ username: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [showSuccessMsg, setShowSuccessMsg] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    setSuccess(false);
    setShowSuccessMsg(false);

    try {
      const res = await fetch('http://localhost:5050/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      const data = await res.json();

      if (res.ok) {
        setSuccess(true);

        // ⏱️ Show success message after animation delay
        setTimeout(() => {
          setShowSuccessMsg(true);
        }, 3000); // After 3 seconds

        // ⏱️ Redirect to login after full delay
        setTimeout(() => {
          navigate('/login');
        }, 5000); // After 5 seconds
      } else {
        setError(data.error || 'Signup failed');
        setLoading(false);
      }
    } catch (err) {
      setError('Something went wrong');
      setLoading(false);
    }
  };

  return (
    <>
      {success && (
        <div className="loading-screen">
          {!showSuccessMsg ? (
            <div className="loading-text">
              {'Creating your Lip Buddy account...'.split('').map((char, i) => (
                <span key={i} style={{ animationDelay: `${i * 0.05}s` }}>{char}</span>
              ))}
            </div>
          ) : (
            <div className="success-text">✅ Account created successfully!</div>
          )}
        </div>
      )}

      <div className="auth-container">
        <h2 className="signup-title">Create Account</h2>

        <input
          name="username"
          placeholder="Username"
          value={formData.username}
          onChange={handleChange}
          disabled={loading || success}
        />

        <input
          name="password"
          type="password"
          placeholder="Password"
          value={formData.password}
          onChange={handleChange}
          disabled={loading || success}
        />

        {!success && error && <p className="error">{error}</p>}

        <button className="fancy-button" onClick={handleSubmit} disabled={loading || success}>
          {loading ? 'Please wait...' : 'Sign Up'}
        </button>

        <button className="link-button" onClick={() => navigate('/login')} disabled={loading || success}>
          Already have an account? Login
        </button>
      </div>
    </>
  );
};

export default Signup;