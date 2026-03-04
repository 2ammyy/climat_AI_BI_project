# backend/analyze_neighbor_impact.py

import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

print("="*60)
print("🔍 DETAILED NEIGHBOR INFLUENCE ANALYSIS")
print("="*60)

# Load the model and data
model = joblib.load('mlartifacts/model_with_neighbors.pkl')
tunisia_df = pd.read_csv('data/merged_data_clean.csv')
neighbor_df = pd.read_csv('data/neighbor_data/neighbor_weather_latest.csv')

# Calculate neighbor influences per region (same as before)
influence_zones = {
    'northwest': ['Jendouba', 'Beja', 'Kef', 'Siliana'],
    'northeast': ['Bizerte', 'Tunis', 'Ariana', 'Ben Arous', 'Manouba', 'Nabeul', 'Zaghouan'],
    'southwest': ['Tozeur', 'Kebili', 'Tataouine', 'Gafsa'],
    'southeast': ['Medenine', 'Gabes', 'Sfax'],
    'coastal': ['Sousse', 'Monastir', 'Mahdia', 'Sfax', 'Gabes'],
    'central': ['Kairouan', 'Kasserine', 'Sidi Bouzid']
}

# Analyze which regions benefit most from neighbor data
print("\n📊 Regional Impact Analysis:")
region_improvements = {}

for region, governorates in influence_zones.items():
    region_data = tunisia_df[tunisia_df['city'].isin(governorates)]
    print(f"\n📍 {region.upper()} ({len(region_data)} samples):")
    
    # Identify influencing neighbors
    influencers = []
    if region in ['northwest', 'central']:
        influencers.append('Algeria')
    if region in ['southwest', 'southeast']:
        influencers.append('Libya')
    if region in ['northeast', 'coastal']:
        influencers.append('Italy/Malta')
    
    print(f"   Influenced by: {', '.join(influencers)}")
    
    # Show current neighbor conditions
    if 'Algeria' in str(influencers):
        algeria_temp = neighbor_df[neighbor_df['country']=='algeria']['temperature'].mean()
        print(f"   🇩🇿 Algeria: {algeria_temp:.1f}°C")
    if 'Libya' in str(influencers):
        libya_temp = neighbor_df[neighbor_df['country']=='libya']['temperature'].mean()
        print(f"   🇱🇾 Libya: {libya_temp:.1f}°C")
    if 'Italy/Malta' in str(influencers):
        italy_temp = neighbor_df[neighbor_df['country']=='italy']['temperature'].mean()
        malta_temp = neighbor_df[neighbor_df['country']=='malta']['temperature'].mean()
        print(f"   🇮🇹 Italy: {italy_temp:.1f}°C, 🇲🇹 Malta: {malta_temp:.1f}°C")

# Correlation analysis
print("\n📈 Neighbor Feature Correlations with Tunisian Weather:")
neighbor_cols = [col for col in tunisia_df.columns if 'influence' in col]
if len(neighbor_cols) > 0:
    corr_matrix = tunisia_df[neighbor_cols + ['temp_avg', 'wind_speed']].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation: Neighbor Features vs Tunisian Weather')
    plt.tight_layout()
    plt.savefig('eda_results/neighbor_correlations.png')
    plt.show()
    
    print("\nTop correlations with Tunisian temperature:")
    temp_corr = corr_matrix['temp_avg'].drop('temp_avg').sort_values(ascending=False)
    print(temp_corr.head())
    
    print("\nTop correlations with Tunisian wind:")
    wind_corr = corr_matrix['wind_speed'].drop('wind_speed').sort_values(ascending=False)
    print(wind_corr.head())

print("\n✅ Analysis complete! Check eda_results/neighbor_correlations.png")