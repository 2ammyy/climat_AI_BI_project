# backend/debug_api.py

import requests
import json

print("🔍 Debugging API...\n")

# Test with minimal data first
test_data = {
    "temp_max": 22.0,
    "temp_min": 15.0,
    "temp_avg": 18.0,
    "wind_speed": 15.0,
    "humidity": 65.0,
    "city": "Tunis"
}

print(f"📤 Sending: {test_data}")

try:
    response = requests.post(
        "http://localhost:8000/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"📥 Status Code: {response.status_code}")
    
    if response.status_code == 500:
        print("\n❌ Server Error Details:")
        print(response.text)  # This will show the actual error
    else:
        print(f"✅ Response: {response.json()}")
        
except Exception as e:
    print(f"❌ Connection Error: {e}")