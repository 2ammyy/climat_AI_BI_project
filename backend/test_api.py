# backend/test_api.py

import requests
import json

# Base URL
base_url = "http://localhost:8000"

# Test health endpoint
print("🔍 Testing Health Endpoint...")
response = requests.get(f"{base_url}/health")
print(f"Health: {response.json()}\n")

# Test risk-info endpoint
print("🔍 Testing Risk Info...")
response = requests.get(f"{base_url}/risk-info")
risk_info = response.json()
print("Risk Levels:")
for level in risk_info['levels']:
    print(f"  {level['color']} {level['name']}: {level['description']}")
print()

# Test prediction endpoint
print("🔍 Testing Prediction...")
test_cases = [
    {
        "name": "Normal Day in Tunis",
        "data": {"temp_max": 22, "temp_min": 15, "temp_avg": 18, "wind_speed": 15, "humidity": 65, "city": "Tunis"}
    },
    {
        "name": "Hot Day in Sfax",
        "data": {"temp_max": 38, "temp_min": 24, "temp_avg": 31, "wind_speed": 20, "humidity": 40, "city": "Sfax"}
    },
    {
        "name": "Windy Day in Bizerte",
        "data": {"temp_max": 24, "temp_min": 18, "temp_avg": 21, "wind_speed": 55, "humidity": 70, "city": "Bizerte"}
    }
]

for test in test_cases:
    print(f"\n📌 {test['name']}:")
    response = requests.post(f"{base_url}/predict", json=test['data'])
    
    if response.status_code == 200:
        result = response.json()
        print(f"  Risk: {result['risk_level']}")
        print(f"  Confidence: {result['confidence']:.1f}%")
        print(f"  Probabilities: {result['probabilities']}")
    else:
        print(f"  Error: {response.status_code}")

print("\n✅ Test complete!")