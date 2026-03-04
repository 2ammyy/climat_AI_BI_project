# backend/neighbor_data_collector.py

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
import json
import os
import time
from pathlib import Path

# Set up MLflow with SQLite backend (better than filesystem)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

class NeighborDataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.weather_api_url = "https://api.openweathermap.org/data/2.5/weather"
        self.forecast_api_url = "https://api.openweathermap.org/data/2.5/forecast"
        
        # Key locations in neighboring countries
        self.locations = {
            'algeria': {
                'annaba': {'lat': 36.9167, 'lon': 7.7667},
                'algiers': {'lat': 36.7538, 'lon': 3.0588},
                'constantine': {'lat': 36.3650, 'lon': 6.6147}
            },
            'libya': {
                'tripoli': {'lat': 32.8872, 'lon': 13.1917},
                'benghazi': {'lat': 32.1167, 'lon': 20.0667}
            },
            'italy': {
                'palermo': {'lat': 38.1157, 'lon': 13.3615},
                'lampedusa': {'lat': 35.5164, 'lon': 12.6081}
            },
            'malta': {
                'valletta': {'lat': 35.8989, 'lon': 14.5146}
            }
        }
        
        # Influence mapping to Tunisian governorates
        self.influence_zones = {
            'northwest': ['Jendouba', 'Beja', 'Kef', 'Siliana'],
            'northeast': ['Bizerte', 'Tunis', 'Ariana', 'Ben Arous', 'Manouba', 'Nabeul', 'Zaghouan'],
            'southwest': ['Tozeur', 'Kebili', 'Tataouine', 'Gafsa'],
            'southeast': ['Medenine', 'Gabes', 'Sfax'],
            'coastal': ['Sousse', 'Monastir', 'Mahdia', 'Sfax', 'Gabes'],
            'central': ['Kairouan', 'Kasserine', 'Sidi Bouzid']
        }
        
        # Create data directory if it doesn't exist
        Path("data/neighbor_data").mkdir(parents=True, exist_ok=True)

    def fetch_neighbor_weather(self, country, city, coords, retries=3):
        """Fetch current weather from neighbor country with retry logic"""
        params = {
            'lat': coords['lat'],
            'lon': coords['lon'],
            'appid': self.api_key,
            'units': 'metric'
        }
        
        for attempt in range(retries):
            try:
                response = requests.get(self.weather_api_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'country': country,
                        'city': city,
                        'temperature': round(data['main']['temp'], 1),
                        'feels_like': round(data['main']['feels_like'], 1),
                        'humidity': data['main']['humidity'],
                        'pressure': data['main']['pressure'],
                        'wind_speed': round(data['wind']['speed'] * 3.6, 1),  # Convert to km/h
                        'wind_direction': data['wind'].get('deg', 0),
                        'weather_condition': data['weather'][0]['main'],
                        'weather_description': data['weather'][0]['description'],
                        'clouds': data['clouds']['all'],
                        'timestamp': datetime.now().isoformat(),
                        'collection_date': datetime.now().strftime('%Y-%m-%d')
                    }
                elif response.status_code == 401:
                    print(f"  ❌ API Key error for {country}-{city}: Invalid key")
                    return None
                else:
                    print(f"  ⚠️ Attempt {attempt+1} failed for {country}-{city}: {response.status_code}")
                    time.sleep(2)  # Wait before retry
                    
            except requests.exceptions.Timeout:
                print(f"  ⚠️ Timeout for {country}-{city}, retrying...")
                time.sleep(2)
            except requests.exceptions.ConnectionError:
                print(f"  ⚠️ Connection error for {country}-{city}, retrying...")
                time.sleep(2)
            except Exception as e:
                print(f"  ⚠️ Error for {country}-{city}: {str(e)}")
                time.sleep(2)
        
        return None

    def collect_all_neighbor_data(self):
        """Collect data from all neighbor locations"""
        all_data = []
        successful = 0
        failed = 0
        
        print("\n" + "="*60)
        print("📡 COLLECTING NEIGHBOR COUNTRY WEATHER DATA")
        print("="*60)
        
        # Create a new MLflow run for this collection
        experiment_name = "neighbor_data_collection"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"collection_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_param("collection_time", datetime.now().isoformat())
            mlflow.log_param("api_key_valid", bool(self.api_key))
            
            for country, locations in self.locations.items():
                print(f"\n📌 {country.upper()}:")
                for city, coords in locations.items():
                    print(f"  🔍 Fetching {city}...", end=" ")
                    
                    data = self.fetch_neighbor_weather(country, city, coords)
                    
                    if data:
                        all_data.append(data)
                        successful += 1
                        print(f"✅ {data['temperature']}°C, wind {data['wind_speed']} km/h")
                        
                        # Log individual metrics
                        mlflow.log_metric(f"{country}_{city}_temp", data['temperature'])
                        mlflow.log_metric(f"{country}_{city}_wind", data['wind_speed'])
                    else:
                        failed += 1
                        print("❌ Failed")
                    
                    # Small delay to avoid rate limiting
                    time.sleep(1)
            
            # Log summary metrics
            mlflow.log_metric("total_successful", successful)
            mlflow.log_metric("total_failed", failed)
            
            # Save to CSV if we have data
            if all_data:
                df = pd.DataFrame(all_data)
                
                # Save with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                csv_path = f"data/neighbor_data/neighbor_weather_{timestamp}.csv"
                df.to_csv(csv_path, index=False)
                
                # Also save as latest
                latest_path = "data/neighbor_data/neighbor_weather_latest.csv"
                df.to_csv(latest_path, index=False)
                
                mlflow.log_artifact(csv_path)
                mlflow.log_artifact(latest_path)
                
                print(f"\n✅ Collection complete! Saved {len(df)} records to {csv_path}")
                
                # Show summary by country
                print("\n📊 Summary by Country:")
                summary = df.groupby('country').agg({
                    'temperature': 'mean',
                    'wind_speed': 'mean',
                    'humidity': 'mean'
                }).round(1)
                print(summary)
                
                return df
            else:
                print("\n❌ No data collected successfully!")
                return None

    def analyze_influence(self, tunisia_region, neighbor_data):
        """Analyze how neighbor weather affects Tunisian regions"""
        region_governorates = self.influence_zones.get(tunisia_region, [])
        
        if not region_governorates:
            return {"error": f"Region '{tunisia_region}' not found"}
        
        analysis = {
            'region': tunisia_region,
            'governorates': region_governorates,
            'neighbor_influences': {},
            'risk_factors': []
        }
        
        # Define which neighbors influence which regions
        influence_map = {
            'northwest': ['algeria'],
            'southwest': ['algeria', 'libya'],
            'northeast': ['italy', 'malta'],
            'southeast': ['libya'],
            'coastal': ['italy', 'malta'],
            'central': ['algeria']
        }
        
        relevant_countries = influence_map.get(tunisia_region, [])
        
        for country in relevant_countries:
            country_data = [d for d in neighbor_data if d['country'] == country]
            if country_data:
                avg_temp = np.mean([d['temperature'] for d in country_data])
                avg_wind = np.mean([d['wind_speed'] for d in country_data])
                avg_humidity = np.mean([d['humidity'] for d in country_data])
                
                analysis['neighbor_influences'][country] = {
                    'avg_temperature': round(avg_temp, 1),
                    'avg_wind_speed': round(avg_wind, 1),
                    'avg_humidity': round(avg_humidity, 1),
                    'data_points': len(country_data)
                }
                
                # Identify risk factors
                if avg_temp > 35:
                    analysis['risk_factors'].append(f"Heat influence from {country}")
                if avg_wind > 60:
                    analysis['risk_factors'].append(f"Wind influence from {country}")
        
        return analysis

    def get_influence_for_governorate(self, governorate, neighbor_data):
        """Get neighbor influence for a specific Tunisian governorate"""
        
        # Determine region based on governorate
        region_map = {}
        for region, govs in self.influence_zones.items():
            for gov in govs:
                region_map[gov] = region
        
        region = region_map.get(governorate, 'central')
        return self.analyze_influence(region, neighbor_data)

    def generate_influence_features(self, tunisia_df, neighbor_df):
        """Generate features for model training based on neighbor influence"""
        
        feature_df = tunisia_df.copy()
        
        # Add empty influence columns
        feature_df['algeria_temp_influence'] = 0.0
        feature_df['libya_temp_influence'] = 0.0
        feature_df['italy_temp_influence'] = 0.0
        feature_df['malta_temp_influence'] = 0.0
        feature_df['algeria_wind_influence'] = 0.0
        feature_df['italy_wind_influence'] = 0.0
        
        # Calculate average conditions for each neighbor
        neighbor_avgs = {}
        for country in ['algeria', 'libya', 'italy', 'malta']:
            country_data = neighbor_df[neighbor_df['country'] == country]
            if len(country_data) > 0:
                neighbor_avgs[country] = {
                    'temp': country_data['temperature'].mean(),
                    'wind': country_data['wind_speed'].mean()
                }
        
        # Apply influences based on region
        for region, governorates in self.influence_zones.items():
            mask = feature_df['city'].isin(governorates)
            
            if region in ['northwest', 'central'] and 'algeria' in neighbor_avgs:
                feature_df.loc[mask, 'algeria_temp_influence'] = neighbor_avgs['algeria']['temp']
                feature_df.loc[mask, 'algeria_wind_influence'] = neighbor_avgs['algeria']['wind']
            
            if region in ['southwest', 'southeast'] and 'libya' in neighbor_avgs:
                feature_df.loc[mask, 'libya_temp_influence'] = neighbor_avgs['libya']['temp']
            
            if region in ['northeast', 'coastal'] and 'italy' in neighbor_avgs:
                feature_df.loc[mask, 'italy_temp_influence'] = neighbor_avgs['italy']['temp']
                feature_df.loc[mask, 'italy_wind_influence'] = neighbor_avgs['italy']['wind']
            
            if region in ['coastal'] and 'malta' in neighbor_avgs:
                feature_df.loc[mask, 'malta_temp_influence'] = neighbor_avgs['malta']['temp']
        
        return feature_df

def main():
    """Main execution function"""
    
    # Your OpenWeatherMap API key
    API_KEY = "139fef2236c773191352b491bd53a624"
    
    # Create collector instance
    collector = NeighborDataCollector(API_KEY)
    
    # Collect fresh data
    print("\n🚀 Starting neighbor weather data collection...")
    neighbor_df = collector.collect_all_neighbor_data()
    
    if neighbor_df is not None and len(neighbor_df) > 0:
        print("\n" + "="*60)
        print("📊 SAMPLE OF COLLECTED DATA")
        print("="*60)
        print(neighbor_df[['country', 'city', 'temperature', 'wind_speed', 'weather_condition']].head(10))
        
        # Analyze influence for each region
        print("\n" + "="*60)
        print("🌍 REGIONAL INFLUENCE ANALYSIS")
        print("="*60)
        
        for region in collector.influence_zones.keys():
            analysis = collector.analyze_influence(region, neighbor_df.to_dict('records'))
            print(f"\n📌 {region.upper()} Region:")
            print(f"   Governorates: {', '.join(analysis['governorates'][:5])}...")
            if analysis['neighbor_influences']:
                for country, stats in analysis['neighbor_influences'].items():
                    print(f"   • {country}: {stats['avg_temperature']}°C, wind {stats['avg_wind_speed']} km/h")
            if analysis['risk_factors']:
                for risk in analysis['risk_factors']:
                    print(f"   ⚠️ {risk}")
        
        # Save influence analysis
        analysis_report = {
            'collection_time': datetime.now().isoformat(),
            'total_locations': len(neighbor_df),
            'countries': neighbor_df['country'].value_counts().to_dict(),
            'regional_analysis': {}
        }
        
        for region in collector.influence_zones.keys():
            analysis_report['regional_analysis'][region] = collector.analyze_influence(
                region, neighbor_df.to_dict('records')
            )
        
        # Save as JSON
        report_path = f"data/neighbor_data/influence_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_path, 'w') as f:
            json.dump(analysis_report, f, indent=2)
        print(f"\n✅ Analysis report saved to {report_path}")
        
        # Demonstrate feature generation
        print("\n" + "="*60)
        print("🔧 GENERATING ENHANCED FEATURES FOR MODEL TRAINING")
        print("="*60)
        
        # Load sample Tunisian data (if available)
        try:
            tunisia_sample = pd.read_csv('data/merged_data_clean.csv').head(10)
            enhanced_features = collector.generate_influence_features(tunisia_sample, neighbor_df)
            print("\n✅ Enhanced features with neighbor influence:")
            print(enhanced_features[['city', 'algeria_temp_influence', 'italy_wind_influence']].head())
        except FileNotFoundError:
            print("\n⚠️ Tunisian data not found. Run this after collecting neighbor data and before training.")
        
        print("\n" + "="*60)
        print("✅ NEIGHBOR DATA COLLECTION COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run MLflow UI: mlflow ui --port 5000")
        print("2. Train enhanced model with neighbor features")
        print("3. Compare performance in MLflow dashboard")
        
    else:
        print("\n❌ Failed to collect neighbor data. Check your API key and internet connection.")

if __name__ == "__main__":
    main()