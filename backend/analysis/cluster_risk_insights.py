# backend/analysis/cluster_risk_insights_fixed.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("📊 Loading your real data...")
df = pd.read_csv('data/merged_preprocessed.csv')

print(f"\n✅ Data loaded! Shape: {df.shape}")
print(f"\n📋 Danger label distribution:")
print(df['danger_label'].value_counts().sort_index())

# Select features
feature_cols = [
    'temp_max',
    'wind_speed_kmh_hist',
    'precip_daily',
    'humidity_percent_hist',
    'sealevelpressure'
]

# Check available features
available_features = [col for col in feature_cols if col in df.columns]
print(f"\n🔍 Available features: {available_features}")

# Prepare features
X = df[available_features].copy()

# Handle missing values
print("\n🔧 Handling missing values...")
print(f"Missing values before:")
print(X.isnull().sum())

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print(f"\nMissing values after imputation:")
print(pd.DataFrame(X_imputed).isnull().sum())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Find natural clusters
print("\n🔍 Finding natural weather clusters...")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['weather_cluster'] = kmeans.fit_predict(X_scaled)

# Analyze each cluster
print("\n📈 Weather Cluster Profiles:")
for cluster_id in range(5):
    cluster_data = df[df['weather_cluster'] == cluster_id]
    print(f"\n📌 Cluster {cluster_id}:")
    print(f"   Size: {len(cluster_data)} ({len(cluster_data)/len(df)*100:.1f}%)")
    for col in available_features[:3]:  # Show main weather features
        print(f"   Avg {col}: {cluster_data[col].mean():.1f}")
    print(f"   Danger labels in this cluster: {cluster_data['danger_label'].value_counts().to_dict()}")

# Create risk levels based on cluster severity
print("\n🎯 Creating risk levels from clusters...")

# Calculate cluster severity
cluster_severity = {}
for cluster_id in range(5):
    cluster_data = df[df['weather_cluster'] == cluster_id]
    severity = (
        cluster_data[available_features[0]].mean() / 50 * 30 +  # temp
        cluster_data[available_features[1]].mean() / 100 * 30 +  # wind
        cluster_data[available_features[2]].mean() / 100 * 40    # rain
    )
    cluster_severity[cluster_id] = severity

# Sort clusters by severity
sorted_clusters = sorted(cluster_severity.items(), key=lambda x: x[1])

# Map to risk levels
risk_map = {
    sorted_clusters[0][0]: 'Green',
    sorted_clusters[1][0]: 'Yellow',
    sorted_clusters[2][0]: 'Orange',
    sorted_clusters[3][0]: 'Red',
    sorted_clusters[4][0]: 'Purple'
}

print("\n📊 Cluster to Risk Level Mapping:")
for cluster, risk in risk_map.items():
    print(f"  Cluster {cluster} (severity: {cluster_severity[cluster]:.2f}) -> {risk}")

# Apply mapping
df['generated_risk'] = df['weather_cluster'].map(risk_map)

# Show final distribution
print("\n📈 Generated Risk Distribution:")
print(df['generated_risk'].value_counts())

# Save the enhanced dataset
df.to_csv('data/merged_data_with_clusters.csv', index=False)
print("\n✅ Enhanced dataset saved to: data/merged_data_with_clusters.csv")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Current danger labels (all 0)
ax = axes[0]
df['danger_label'].value_counts().plot(kind='bar', ax=ax, color='green')
ax.set_title('Current Danger Labels (All Green)')
ax.set_xlabel('Danger Label')
ax.set_ylabel('Count')

# Generated risk levels
ax = axes[1]
colors = {'Green': 'green', 'Yellow': 'yellow', 'Orange': 'orange', 'Red': 'red', 'Purple': 'purple'}
df['generated_risk'].value_counts().sort_index().plot(kind='bar', ax=ax, 
                                                      color=[colors[x] for x in df['generated_risk'].value_counts().sort_index().index])
ax.set_title('Generated Risk Levels from Weather Clusters')
ax.set_xlabel('Risk Level')
ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('eda_results/risk_generation.png')
plt.show()

print("\n✅ Analysis complete!")