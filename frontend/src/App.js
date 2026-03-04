// frontend/src/App.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [forecastData, setForecastData] = useState({
    date: new Date().toISOString().split('T')[0],
    city: 'Tunis'
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [governorates, setGovernorates] = useState([]);
  const [apiStatus, setApiStatus] = useState('checking');
  const [apiUrl, setApiUrl] = useState('');

  const possibleUrls = [
    'http://localhost:8000',
    'http://127.0.0.1:8000'
  ];

  // Check API connection on load
  useEffect(() => {
    findWorkingApiUrl();
    fetchGovernorates();
  }, []);

  const findWorkingApiUrl = async () => {
    for (const url of possibleUrls) {
      try {
        console.log(`Trying to connect to ${url}...`);
        const response = await axios.get(`${url}/health`, { timeout: 2000 });
        
        if (response.status === 200) {
          console.log(`✅ Connected to ${url}`);
          setApiUrl(url);
          setApiStatus('connected');
          return;
        }
      } catch (err) {
        console.log(`❌ Failed to connect to ${url}:`, err.message);
      }
    }
    setApiStatus('disconnected');
  };

  const fetchGovernorates = async () => {
    try {
      // Try to fetch from API first
      if (apiUrl) {
        const response = await axios.get(`${apiUrl}/governorates`);
        setGovernorates(response.data.governorates);
      } else {
        // Fallback to hardcoded list
        setGovernorates([
          "Tunis", "Ariana", "Ben Arous", "Manouba", "Nabeul", "Zaghouan",
          "Bizerte", "Beja", "Jendouba", "Kef", "Siliana", "Sousse",
          "Monastir", "Mahdia", "Sfax", "Kairouan", "Kasserine", "Sidi Bouzid",
          "Gabes", "Medenine", "Tataouine", "Gafsa", "Tozeur", "Kebili"
        ]);
      }
    } catch (err) {
      console.error('Error fetching governorates:', err);
      // Fallback governorates
      setGovernorates(['Tunis', 'Sfax', 'Sousse', 'Bizerte', 'Jendouba']);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setForecastData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);
    
    try {
      console.log('📤 Sending request:', forecastData);
      
      const response = await axios.post(`${apiUrl}/forecast-by-date`, {
        date: forecastData.date,
        city: forecastData.city
      });
      
      console.log('📥 Received response:', response.data);
      setPrediction(response.data);
      
    } catch (err) {
      console.error('❌ Error:', err);
      
      let errorMessage = 'Error getting forecast';
      
      if (err.code === 'ECONNREFUSED' || err.message.includes('Network Error')) {
        errorMessage = 'Cannot connect to backend server. Make sure it is running.';
      } else if (err.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        if (err.response.status === 404) {
          errorMessage = `No forecast available for ${forecastData.date}. Try a different date.`;
        } else if (err.response.status === 400) {
          errorMessage = err.response.data?.detail || 'Invalid request. Please check your input.';
        } else if (err.response.status === 502) {
          errorMessage = 'Weather service unavailable. Please try again later.';
        } else {
          errorMessage = err.response.data?.detail || `Server error: ${err.response.status}`;
        }
      } else if (err.request) {
        // The request was made but no response was received
        errorMessage = 'No response from server. Check if backend is running.';
      } else {
        // Something happened in setting up the request that triggered an Error
        errorMessage = err.message;
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (riskLevel) => {
    const colors = {
      'GREEN': '#4CAF50',
      'YELLOW': '#FFC107',
      'ORANGE': '#FF9800',
      'RED': '#F44336',
      'PURPLE': '#9C27B0'
    };
    return colors[riskLevel] || '#999';
  };

  const getRiskEmoji = (riskLevel) => {
    const emojis = {
      'GREEN': '✅',
      'YELLOW': '⚠️',
      'ORANGE': '⚡',
      'RED': '🔴',
      'PURPLE': '🟣'
    };
    return emojis[riskLevel] || '❓';
  };

  // Format date for display
  const formatDate = (dateString) => {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(dateString).toLocaleDateString('en-US', options);
  };

  // Get min and max dates (today to 5 days ahead)
  const today = new Date();
  const minDate = today.toISOString().split('T')[0];
  const maxDate = new Date(today.setDate(today.getDate() + 5)).toISOString().split('T')[0];

  return (
    <div className="App">
      <header className="app-header">
        <h1>🌤️ WeatherGuardTN</h1>
        <p>Tunisia Weather Danger & Vigilance Predictor</p>
        
        {/* API Status Indicator */}
        <div className={`api-status ${apiStatus}`}>
          {apiStatus === 'connected' && (
            <div className="status-success">
              ✅ Connected to Weather Service
            </div>
          )}
          {apiStatus === 'checking' && (
            <div className="status-checking">
              ⏳ Checking connection...
            </div>
          )}
          {apiStatus === 'disconnected' && (
            <div className="status-error">
              ❌ Cannot connect to backend. Please start the server.
            </div>
          )}
        </div>
      </header>

      <div className="main-container" style={{ maxWidth: '600px', margin: '0 auto' }}>
        {/* Input Form - Only Date and City */}
        <div className="input-section">
          <h2>Weather Risk Forecast</h2>
          <p className="form-description">
            Select a date and city to get the weather forecast and risk prediction
          </p>
          
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="date">📅 Select Date:</label>
              <input
                type="date"
                id="date"
                name="date"
                value={forecastData.date}
                onChange={handleInputChange}
                min={minDate}
                max={maxDate}
                required
              />
              <small className="field-hint">Choose a date up to 5 days ahead</small>
            </div>

            <div className="form-group">
              <label htmlFor="city">📍 Select City:</label>
              <select 
                id="city"
                name="city" 
                value={forecastData.city} 
                onChange={handleInputChange}
                required
              >
                {governorates.map(city => (
                  <option key={city} value={city}>{city}</option>
                ))}
              </select>
            </div>

            <button 
              type="submit" 
              disabled={loading || apiStatus !== 'connected'} 
              className="predict-btn"
            >
              {loading ? (
                <span>⏳ Getting Forecast...</span>
              ) : (
                <span>🔮 Get Risk Prediction</span>
              )}
            </button>
          </form>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="loading-section">
            <div className="loading-spinner"></div>
            <p>Fetching weather data and predicting risk...</p>
          </div>
        )}

        {/* Prediction Result */}
        {prediction && !loading && (
          <div className="result-section">
            <h2>Forecast Results</h2>
            <div className="result-date">
              {formatDate(prediction.forecast_date)} - {prediction.city}
            </div>
            
            <div 
              className="result-card" 
              style={{ backgroundColor: getRiskColor(prediction.risk_level) }}
            >
              <div className="result-header">
                <span className="result-emoji">{getRiskEmoji(prediction.risk_level)}</span>
                <span className="result-risk">{prediction.risk_level}</span>
              </div>
              
              <div className="result-details">
                {/* Weather Details */}
                <div className="weather-details">
                  <h4>🌡️ Weather Conditions:</h4>
                  <div className="weather-grid">
                    <div className="weather-item">
                      <span className="weather-label">Max Temp:</span>
                      <span className="weather-value">{prediction.weather.temp_max}°C</span>
                    </div>
                    <div className="weather-item">
                      <span className="weather-label">Min Temp:</span>
                      <span className="weather-value">{prediction.weather.temp_min}°C</span>
                    </div>
                    <div className="weather-item">
                      <span className="weather-label">Avg Temp:</span>
                      <span className="weather-value">{prediction.weather.temp_avg}°C</span>
                    </div>
                    <div className="weather-item">
                      <span className="weather-label">Wind Speed:</span>
                      <span className="weather-value">{prediction.weather.wind_speed} km/h</span>
                    </div>
                    <div className="weather-item">
                      <span className="weather-label">Humidity:</span>
                      <span className="weather-value">{prediction.weather.humidity}%</span>
                    </div>
                  </div>
                  <div className="confidence-badge">
                    Confidence: {prediction.confidence}%
                  </div>
                </div>

                {/* Probability Distribution */}
                <div className="probabilities">
                  <h4>📊 Risk Probabilities:</h4>
                  {Object.entries(prediction.probabilities).map(([risk, prob]) => (
                    <div key={risk} className="probability-bar">
                      <div className="probability-label">
                        <span>{risk}</span>
                        <span>{prob}%</span>
                      </div>
                      <div className="bar-container">
                        <div 
                          className="bar" 
                          style={{ 
                            width: `${prob}%`,
                            backgroundColor: getRiskColor(risk)
                          }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Action Recommendations */}
            <div className="recommendations">
              <h4>Recommended Actions:</h4>
              <ul>
                {prediction.risk_level === 'GREEN' && (
                  <>
                    <li>✅ Normal conditions - No action needed</li>
                    <li>✅ Regular activities can continue</li>
                  </>
                )}
                {prediction.risk_level === 'YELLOW' && (
                  <>
                    <li>⚠️ Be aware of weather conditions</li>
                    <li>⚠️ Monitor local weather updates</li>
                    <li>⚠️ Plan outdoor activities with caution</li>
                  </>
                )}
                {prediction.risk_level === 'ORANGE' && (
                  <>
                    <li>⚡ Be prepared for possible disruptions</li>
                    <li>⚡ Secure outdoor objects</li>
                    <li>⚡ Avoid unnecessary travel</li>
                    <li>⚡ Stay informed about weather alerts</li>
                  </>
                )}
                {prediction.risk_level === 'RED' && (
                  <>
                    <li>🔴 Take action to protect life and property</li>
                    <li>🔴 Stay indoors if possible</li>
                    <li>🔴 Follow official instructions</li>
                    <li>🔴 Prepare for emergency supplies</li>
                  </>
                )}
                {prediction.risk_level === 'PURPLE' && (
                  <>
                    <li>🟣 EMERGENCY - Immediate action required</li>
                    <li>🟣 Seek shelter immediately</li>
                    <li>🟣 Follow evacuation orders</li>
                    <li>🟣 Stay tuned to emergency services</li>
                  </>
                )}
              </ul>
            </div>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="error-message">
            <strong>❌ Error:</strong> {error}
            <div className="error-help">
              <p>💡 Troubleshooting tips:</p>
              <ul>
                <li>Make sure the backend server is running:
                  <br />
                  <code>uvicorn backend.app.api.weather_risk:app --reload --port 8000</code>
                </li>
                <li>Check that you're using a valid date (up to 5 days ahead)</li>
                <li>Verify your OpenWeatherMap API key is correct</li>
                <li>Try selecting a different city</li>
              </ul>
            </div>
          </div>
        )}

        {/* API Key Note */}
        <div className="info-note">
          <p>ℹ️ Weather data provided by OpenWeatherMap</p>
        </div>
      </div>
    </div>
  );
}

export default App;