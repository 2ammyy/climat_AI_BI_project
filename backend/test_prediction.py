# backend/test_prediction.py

import requests
import json

# Test your API
base_url = "http://localhost:8000"

# Test health
response = requests.get(f"{base_url}/health")
print(f"Health: {response.json()}")

# Test prediction for Tunis
test_data = {
    "governorate": "Tunis",
    "temperature": 35,
    "wind_speed": 45,
    "precipitation": 20,
    "humidity": 60,
    "pressure": 1015
}

response = requests.post(f"{base_url}/api/v1/predict", json=test_data)
if response.status_code == 200:
    print("\n✅ Prediction successful!")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"\n❌ Error: {response.status_code}")
    print(response.text)