import React, { useState, useEffect, useCallback, useMemo } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  // UI State
  const [currentPage, setCurrentPage] = useState('home');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Prediction Form State
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
  const [riskInfo, setRiskInfo] = useState(null);
  
  // Map data state
  const [mapData, setMapData] = useState(null);
  const [loadingMap, setLoadingMap] = useState(false);

  // Move possibleUrls inside useMemo to prevent recreation on every render
  const possibleUrls = useMemo(() => [
    'http://127.0.0.1:8000',
    'http://localhost:8000'
  ], []);

  // Define functions with useCallback
  const findWorkingApiUrl = useCallback(async () => {
    console.log('🔍 Starting API connection check...');
    for (const url of possibleUrls) {
      try {
        console.log(`🔍 Trying to connect to ${url}/health...`);
        const response = await axios.get(`${url}/health`, { timeout: 2000 });
        
        if (response.status === 200) {
          console.log(`✅ Connected to ${url}`);
          console.log('✅ Response data:', response.data);
          setApiUrl(url);
          setApiStatus('connected');
          return;
        }
      } catch (err) {
        console.log(`❌ Failed to connect to ${url}:`, err.message);
      }
    }
    console.log('❌ All connection attempts failed');
    setApiStatus('disconnected');
  }, [possibleUrls]);

  const fetchGovernorates = useCallback(async () => {
    if (!apiUrl) return;
    
    try {
      console.log('📥 Fetching governorates from:', `${apiUrl}/governorates`);
      const response = await axios.get(`${apiUrl}/governorates`);
      setGovernorates(response.data.governorates);
    } catch (err) {
      console.error('Error fetching governorates:', err);
      // Fallback governorates
      setGovernorates([
        "Tunis", "Ariana", "Ben Arous", "Manouba", "Nabeul", "Zaghouan",
        "Bizerte", "Beja", "Jendouba", "Kef", "Siliana", "Sousse",
        "Monastir", "Mahdia", "Sfax", "Kairouan", "Kasserine", "Sidi Bouzid",
        "Gabes", "Medenine", "Tataouine", "Gafsa", "Tozeur", "Kebili"
      ]);
    }
  }, [apiUrl]);

  const fetchRiskInfo = useCallback(async () => {
    if (!apiUrl) return;
    
    try {
      console.log('📥 Fetching risk info from:', `${apiUrl}/risk-info`);
      const response = await axios.get(`${apiUrl}/risk-info`);
      setRiskInfo(response.data);
    } catch (err) {
      console.error('Error fetching risk info:', err);
    }
  }, [apiUrl]);

  // Generate dynamic map data
  const generateMapData = useCallback(() => {
    const governorateList = [
      "Tunis", "Ariana", "Ben Arous", "Manouba", "Nabeul", "Zaghouan",
      "Bizerte", "Beja", "Jendouba", "Kef", "Siliana", "Sousse",
      "Monastir", "Mahdia", "Sfax", "Kairouan", "Kasserine", "Sidi Bouzid",
      "Gabes", "Medenine", "Tataouine", "Gafsa", "Tozeur", "Kebili"
    ];
    
    const mockData = {};
    governorateList.forEach(gov => {
      const rand = Math.random();
      if (rand < 0.3) mockData[gov] = 'GREEN';
      else if (rand < 0.55) mockData[gov] = 'YELLOW';
      else if (rand < 0.75) mockData[gov] = 'ORANGE';
      else if (rand < 0.9) mockData[gov] = 'RED';
      else mockData[gov] = 'PURPLE';
    });
    
    setMapData(mockData);
    setLoadingMap(false);
  }, []);

  // Debug: Log API URL changes
  useEffect(() => {
    console.log('🔍 Current API URL:', apiUrl);
    console.log('🔍 API Status:', apiStatus);
  }, [apiUrl, apiStatus]);

  // Check API connection on load
  useEffect(() => {
    findWorkingApiUrl();
  }, [findWorkingApiUrl]);

  // Fetch governorates and risk info when API URL changes
  useEffect(() => {
    if (apiUrl && apiStatus === 'connected') {
      fetchGovernorates();
      fetchRiskInfo();
    }
  }, [apiUrl, apiStatus, fetchGovernorates, fetchRiskInfo]);

  // Generate map data
  useEffect(() => {
    setLoadingMap(true);
    generateMapData();
  }, [generateMapData]);

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
      console.log('📤 Sending request to /forecast-by-date:', forecastData);
      console.log('📤 Full URL:', `${apiUrl}/forecast-by-date`);
      
      // Use ONLY the endpoint that exists in your backend
      const response = await axios.post(`${apiUrl}/forecast-by-date`, {
        date: forecastData.date,
        city: forecastData.city
      });
      
      console.log('📥 Received response:', response.data);
      setPrediction(response.data);
      
    } catch (err) {
      console.error('❌ Full error object:', err);
      
      let errorMessage = 'Error getting forecast';
      
      if (err.code === 'ECONNREFUSED' || err.message.includes('Network Error')) {
        errorMessage = 'Cannot connect to backend server. Make sure it is running.';
      } else if (err.response) {
        console.log('❌ Error response status:', err.response.status);
        console.log('❌ Error response data:', err.response.data);
        console.log('❌ Tried to call:', err.config?.url);
        
        if (err.response.status === 404) {
          errorMessage = `No forecast available for ${forecastData.date} in ${forecastData.city}. Try a different date or city.`;
        } else if (err.response.status === 400) {
          errorMessage = err.response.data?.detail || 'Invalid request. Please check your input.';
        } else if (err.response.status === 422) {
          errorMessage = 'Date format error. Please use YYYY-MM-DD format.';
        } else if (err.response.status === 502) {
          errorMessage = 'Weather service unavailable. Please try again later.';
        } else {
          errorMessage = err.response.data?.detail || `Server error: ${err.response.status}`;
        }
      } else if (err.request) {
        console.log('❌ Request made but no response received');
        errorMessage = 'No response from server. Check if backend is running.';
      } else {
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

  const formatDate = (dateString) => {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(dateString).toLocaleDateString('en-US', options);
  };

  const today = new Date();
  const minDate = today.toISOString().split('T')[0];
  const maxDate = new Date(today.setDate(today.getDate() + 5)).toISOString().split('T')[0];

  const getPageSource = () => {
    const basePath = '/eda_results/cross_border/';
    switch(currentPage) {
      case 'dashboard':
        return `${basePath}dashboard.html`;
      case 'scatter':
        return `${basePath}interactive_scatter.html`;
      case 'tempbars':
        return `${basePath}temperature_bars.html`;
      case 'weathermap':
        return `${basePath}weather_map.html`;
      default:
        return '';
    }
  };

  const handleNavClick = (page) => (e) => {
    e.preventDefault();
    setCurrentPage(page);
  };

  // Helper function to get risk level for a governorate
  const getGovRisk = (govName) => {
    if (mapData && mapData[govName]) {
      return mapData[govName].toLowerCase();
    }
    // Default fallback with some variety
    const defaults = {
      'Tunis': 'yellow',
      'Ariana': 'green',
      'Ben Arous': 'green',
      'Manouba': 'green',
      'Nabeul': 'yellow',
      'Zaghouan': 'green',
      'Bizerte': 'yellow',
      'Beja': 'orange',
      'Jendouba': 'orange',
      'Kef': 'yellow',
      'Siliana': 'green',
      'Sousse': 'green',
      'Monastir': 'green',
      'Mahdia': 'yellow',
      'Sfax': 'orange',
      'Kairouan': 'green',
      'Kasserine': 'orange',
      'Sidi Bouzid': 'yellow',
      'Gabes': 'red',
      'Medenine': 'orange',
      'Tataouine': 'purple',
      'Gafsa': 'yellow',
      'Tozeur': 'green',
      'Kebili': 'yellow'
    };
    return defaults[govName] || 'green';
  };

  return (
    <div className="app-wrapper">
      {/* Sidebar */}
      <aside className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
        <div className="sidebar-header">
          <div className="logo-area">
            <i className="fas fa-cloud-sun-rain logo-icon"></i>
            {!sidebarCollapsed && <span className="logo-text">WeatherGuardTN</span>}
          </div>
          <button 
            className="toggle-btn" 
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            aria-label="Toggle sidebar"
          >
            <i className={`fas fa-chevron-${sidebarCollapsed ? 'right' : 'left'}`}></i>
          </button>
        </div>

        <nav className="nav-menu">
          <button 
            className={`nav-item ${currentPage === 'home' ? 'active' : ''}`}
            onClick={handleNavClick('home')}
          >
            <i className="fas fa-home"></i>
            {!sidebarCollapsed && <span>Vigilance Home</span>}
          </button>
          <button 
            className={`nav-item ${currentPage === 'dashboard' ? 'active' : ''}`}
            onClick={handleNavClick('dashboard')}
          >
            <i className="fas fa-chart-pie"></i>
            {!sidebarCollapsed && <span>Risk Dashboard</span>}
          </button>
          <button 
            className={`nav-item ${currentPage === 'scatter' ? 'active' : ''}`}
            onClick={handleNavClick('scatter')}
          >
            <i className="fas fa-dot-circle"></i>
            {!sidebarCollapsed && <span>Multi-risk scatter</span>}
          </button>
          <button 
            className={`nav-item ${currentPage === 'tempbars' ? 'active' : ''}`}
            onClick={handleNavClick('tempbars')}
          >
            <i className="fas fa-chart-bar"></i>
            {!sidebarCollapsed && <span>Temperature bars</span>}
          </button>
          <button 
            className={`nav-item ${currentPage === 'weathermap' ? 'active' : ''}`}
            onClick={handleNavClick('weathermap')}
          >
            <i className="fas fa-map-marked-alt"></i>
            {!sidebarCollapsed && <span>Weather map</span>}
          </button>
        </nav>

        <div className="sidebar-footer">
          <i className="fas fa-umbrella-beach"></i> 
          {!sidebarCollapsed && <span>5‑level vigilance</span>}
        </div>
      </aside>

      {/* Main Content */}
      <main className={`main-content ${sidebarCollapsed ? 'expanded' : ''}`}>
        {currentPage === 'home' ? (
          <div className="home-page">
            <div className="hero-block">
              <h1>Stay ahead of the storm</h1>
              <div className="subhead">
                Tunisia's first hyperlocal danger predictor — for people, authorities & fishermen
              </div>
            </div>

            {/* API Status Indicator */}
            <div className="api-status-container">
              {apiStatus === 'connected' && (
                <div className="status-success chip">
                  <i className="fas fa-check-circle"></i> Connected to Weather Service
                </div>
              )}
              {apiStatus === 'checking' && (
                <div className="status-checking chip">
                  <i className="fas fa-spinner fa-spin"></i> Checking connection...
                </div>
              )}
              {apiStatus === 'disconnected' && (
                <div className="status-error chip">
                  <i className="fas fa-exclamation-triangle"></i> Cannot connect to backend. Please start the server.
                </div>
              )}
            </div>

            {/* User Chips */}
            <div className="user-chips">
              <span className="chip"><i className="fas fa-graduation-cap"></i> Students & Parents</span>
              <span className="chip"><i className="fas fa-truck"></i> Delivery Drivers</span>
              <span className="chip"><i className="fas fa-ship"></i> Fishermen & Mariners</span>
              <span className="chip"><i className="fas fa-users"></i> General Population</span>
              <span className="chip"><i className="fas fa-helmet-safety"></i> Civil Protection</span>
            </div>

            {/* Prediction Form Card */}
            <div className="prediction-form-card">
              <h2>
                <i className="fas fa-cloud-sun-rain"></i> Get Personalized Risk Assessment
              </h2>
              
              <form className="prediction-form" onSubmit={handleSubmit}>
                <div className="form-row">
                  <div className="form-group">
                    <label><i className="fas fa-calendar"></i> Select Date</label>
                    <input
                      type="date"
                      name="date"
                      value={forecastData.date}
                      onChange={handleInputChange}
                      min={minDate}
                      max={maxDate}
                      required
                    />
                    <small className="field-hint">Up to 5 days ahead</small>
                  </div>

                  <div className="form-group">
                    <label><i className="fas fa-map-marker-alt"></i> Select Governorate</label>
                    <select 
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
                </div>

                <button 
                  type="submit" 
                  className="predict-btn"
                  disabled={loading || apiStatus !== 'connected'}
                >
                  {loading ? (
                    <span><i className="fas fa-spinner fa-spin"></i> Getting Forecast...</span>
                  ) : (
                    <span><i className="fas fa-magic"></i> Get Risk Prediction</span>
                  )}
                </button>
              </form>

              {/* Loading State */}
              {loading && (
                <div className="loading-section">
                  <div className="loading-spinner"></div>
                  <p>Fetching weather data and predicting risk...</p>
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="error-message">
                  <strong><i className="fas fa-exclamation-circle"></i> Error:</strong> {error}
                </div>
              )}

              {/* Prediction Result */}
              {prediction && !loading && (
                <div className="prediction-result">
                  <h3>Your Risk Assessment for {formatDate(prediction.forecast_date)}</h3>
                  
                  <div 
                    className="risk-indicator"
                    style={{ 
                      backgroundColor: getRiskColor(prediction.risk_level) + '20',
                      borderColor: getRiskColor(prediction.risk_level),
                      borderWidth: '2px',
                      borderStyle: 'solid'
                    }}
                  >
                    <span className="risk-level">
                      {getRiskEmoji(prediction.risk_level)} {prediction.risk_level}
                    </span>
                    <span className="risk-probability">Confidence: {prediction.confidence}%</span>
                  </div>

                  <div className="weather-details">
                    <h4>🌡️ Weather Conditions in {prediction.city}:</h4>
                    <div className="weather-grid">
                      <div className="weather-item">
                        <span className="weather-label">Max Temperature:</span>
                        <span className="weather-value">{prediction.weather.temp_max}°C</span>
                      </div>
                      <div className="weather-item">
                        <span className="weather-label">Min Temperature:</span>
                        <span className="weather-value">{prediction.weather.temp_min}°C</span>
                      </div>
                      <div className="weather-item">
                        <span className="weather-label">Average Temperature:</span>
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

                  {/* Action Recommendations */}
                  <div className="recommendations">
                    <h4>Recommended Actions:</h4>
                    <ul>
                      {prediction.risk_level === 'GREEN' && (
                        <>
                          <li><i className="fas fa-check-circle" style={{color: '#4CAF50'}}></i> Normal conditions - No action needed</li>
                          <li><i className="fas fa-check-circle" style={{color: '#4CAF50'}}></i> Regular activities can continue</li>
                        </>
                      )}
                      {prediction.risk_level === 'YELLOW' && (
                        <>
                          <li><i className="fas fa-exclamation-triangle" style={{color: '#FFC107'}}></i> Be aware of weather conditions</li>
                          <li><i className="fas fa-exclamation-triangle" style={{color: '#FFC107'}}></i> Monitor local weather updates</li>
                          <li><i className="fas fa-exclamation-triangle" style={{color: '#FFC107'}}></i> Plan outdoor activities with caution</li>
                        </>
                      )}
                      {prediction.risk_level === 'ORANGE' && (
                        <>
                          <li><i className="fas fa-exclamation-circle" style={{color: '#FF9800'}}></i> Be prepared for possible disruptions</li>
                          <li><i className="fas fa-exclamation-circle" style={{color: '#FF9800'}}></i> Secure outdoor objects</li>
                          <li><i className="fas fa-exclamation-circle" style={{color: '#FF9800'}}></i> Avoid unnecessary travel</li>
                          <li><i className="fas fa-exclamation-circle" style={{color: '#FF9800'}}></i> Stay informed about weather alerts</li>
                        </>
                      )}
                      {prediction.risk_level === 'RED' && (
                        <>
                          <li><i className="fas fa-times-circle" style={{color: '#F44336'}}></i> Take action to protect life and property</li>
                          <li><i className="fas fa-times-circle" style={{color: '#F44336'}}></i> Stay indoors if possible</li>
                          <li><i className="fas fa-times-circle" style={{color: '#F44336'}}></i> Follow official instructions</li>
                          <li><i className="fas fa-times-circle" style={{color: '#F44336'}}></i> Prepare for emergency supplies</li>
                        </>
                      )}
                      {prediction.risk_level === 'PURPLE' && (
                        <>
                          <li><i className="fas fa-skull-crosswind" style={{color: '#9C27B0'}}></i> EMERGENCY - Immediate action required</li>
                          <li><i className="fas fa-skull-crosswind" style={{color: '#9C27B0'}}></i> Seek shelter immediately</li>
                          <li><i className="fas fa-skull-crosswind" style={{color: '#9C27B0'}}></i> Follow evacuation orders</li>
                          <li><i className="fas fa-skull-crosswind" style={{color: '#9C27B0'}}></i> Stay tuned to emergency services</li>
                        </>
                      )}
                    </ul>
                  </div>
                </div>
              )}
            </div>

            {/* Vigilance Map Card - Dynamic Version */}
           {/* I have removed the dynamic vigilance map card  for the moment , It will be added back later  */}

            {/* Risk Info */}
            {riskInfo && (
              <div className="risk-info-card">
                <h4><i className="fas fa-info-circle"></i> Risk Levels Explained:</h4>
                <div className="risk-levels-grid">
                  {riskInfo.levels.map((level) => (
                    <div key={level.code} className="risk-level-item">
                      <span className="risk-color-dot" style={{ backgroundColor: getRiskColor(level.name) }}></span>
                      <span className="risk-name">{level.name}</span>
                      <span className="risk-description">{level.description}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Info Note */}
            <div className="info-note">
              <p><i className="fas fa-cloud-sun"></i> Weather data provided by OpenWeatherMap · Model: LGBMClassifier</p>
            </div>
          </div>
        ) : (
          <iframe
            src={getPageSource()}
            className="page-iframe"
            title={currentPage}
            frameBorder="0"
          />
        )}

        {/* Footer */}
        <footer className="site-footer">
          <h3><i className="fas fa-paper-plane"></i> Contact us · feedback</h3>
          <form className="footer-form" onSubmit={(e) => e.preventDefault()}>
            <div className="form-group">
              <label><i className="far fa-user"></i> Name</label>
              <input type="text" placeholder="Your name" />
            </div>
            <div className="form-group">
              <label><i className="far fa-envelope"></i> Email</label>
              <input type="email" placeholder="name@example.com" />
            </div>
            <div className="form-group">
              <label><i className="far fa-comment"></i> Message</label>
              <textarea placeholder="Your feedback / danger report..."></textarea>
            </div>
            <button type="submit"><i className="fas fa-feather-alt"></i> Send</button>
          </form>
          <p className="footer-note">
            <i className="fas fa-heart" style={{ color: '#da7b44' }}></i> protecting lives – made in tunisia
          </p>
        </footer>
      </main>
    </div>
  );
}

export default App;