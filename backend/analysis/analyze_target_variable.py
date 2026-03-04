"""
Comprehensive Target Variable Analysis for WeatherGuardTN
Run this script to evaluate your danger risk target variable quality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class TargetVariableAnalyzer:
    """
    Comprehensive analysis of danger risk target variable
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path or "../data/merged_data_with_risk.csv"
        self.df = None
        self.feature_cols = ['temperature', 'wind_speed', 'precipitation', 'humidity', 'pressure']
        self.target_col = 'risk_level'
        self.results = {}
        
    def load_and_prepare_data(self):
        """
        Load your merged dataset
        """
        print("\n" + "="*60)
        print("📊 LOADING DATA")
        print("="*60)
        
        try:
            # Try to load your merged data
            if os.path.exists(self.data_path):
                self.df = pd.read_csv(self.data_path)
                print(f"✅ Loaded data from: {self.data_path}")
            else:
                # If not found, look for any CSV in data directory
                data_dir = Path("../data")
                csv_files = list(data_dir.glob("*.csv"))
                if csv_files:
                    self.df = pd.read_csv(csv_files[0])
                    print(f"✅ Loaded data from: {csv_files[0]}")
                else:
                    # Create sample data for testing (remove this in production)
                    print("⚠️ No data found. Creating sample data for testing...")
                    self._create_sample_data()
            
            print(f"\n📈 Data Shape: {self.df.shape}")
            print(f"📅 Date Range: {self.df['date'].min() if 'date' in self.df.columns else 'N/A'} to {self.df['date'].max() if 'date' in self.df.columns else 'N/A'}")
            print(f"📍 Governorates: {self.df['governorate'].nunique() if 'governorate' in self.df.columns else 'N/A'}")
            
            # Display basic info
            print("\n📋 First few rows:")
            print(self.df.head())
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _create_sample_data(self):
        """
        Create sample data for testing (remove in production)
        """
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic Tunisian weather data
        data = {
            'date': pd.date_range(start='2020-01-01', periods=n_samples, freq='D'),
            'governorate': np.random.choice(['Tunis', 'Sfax', 'Sousse', 'Bizerte', 'Jendouba'], n_samples),
            'temperature': np.random.normal(25, 8, n_samples),  # Mean 25°C, std 8
            'wind_speed': np.random.gamma(2, 10, n_samples),  # Skewed right
            'precipitation': np.random.exponential(5, n_samples),  # Many zeros, some heavy rain
            'humidity': np.random.normal(65, 15, n_samples),
            'pressure': np.random.normal(1013, 10, n_samples)
        }
        
        # Clip to realistic ranges
        data['temperature'] = np.clip(data['temperature'], 0, 50)
        data['wind_speed'] = np.clip(data['wind_speed'], 0, 120)
        data['precipitation'] = np.clip(data['precipitation'], 0, 150)
        data['humidity'] = np.clip(data['humidity'], 0, 100)
        data['pressure'] = np.clip(data['pressure'], 980, 1040)
        
        self.df = pd.DataFrame(data)
        
        # Create risk levels based on weather (simulating your labeling)
        conditions = [
            (self.df['temperature'] > 40) | (self.df['wind_speed'] > 90) | (self.df['precipitation'] > 80),
            (self.df['temperature'] > 35) | (self.df['wind_speed'] > 70) | (self.df['precipitation'] > 50),
            (self.df['temperature'] > 30) | (self.df['wind_speed'] > 50) | (self.df['precipitation'] > 30),
            (self.df['temperature'] > 25) | (self.df['wind_speed'] > 30) | (self.df['precipitation'] > 10),
            (self.df['temperature'] <= 25) & (self.df['wind_speed'] <= 30) & (self.df['precipitation'] <= 10)
        ]
        
        choices = ['Purple', 'Red', 'Orange', 'Yellow', 'Green']
        self.df['risk_level'] = np.select(conditions, choices, default='Green')
        
        print("⚠️ Using SAMPLE DATA for testing. Replace with your real data!")
    
    def analyze_distribution(self):
        """
        Analyze target variable distribution
        """
        print("\n" + "="*60)
        print("📊 TARGET DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # 1. Basic distribution
        dist = self.df[self.target_col].value_counts()
        dist_pct = self.df[self.target_col].value_counts(normalize=True) * 100
        
        print("\n📈 Risk Level Distribution:")
        for risk in ['Green', 'Yellow', 'Orange', 'Red', 'Purple']:
            if risk in dist.index:
                count = dist[risk]
                pct = dist_pct[risk]
                bar = "█" * int(pct/2)
                print(f"  {risk}: {count:5d} ({pct:5.1f}%) {bar}")
            else:
                print(f"  {risk}: {0:5d} ({0:5.1f}%)")
        
        # 2. Check for imbalance
        min_pct = dist_pct.min()
        max_pct = dist_pct.max()
        imbalance_ratio = max_pct / min_pct if min_pct > 0 else float('inf')
        
        print(f"\n⚖️ Imbalance Ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 10:
            print("⚠️ SEVERE IMBALANCE: Some risk levels are very rare")
        elif imbalance_ratio > 5:
            print("⚠️ MODERATE IMBALANCE: Consider resampling techniques")
        else:
            print("✅ Good balance across risk levels")
        
        # 3. Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Bar plot
        colors = ['green', 'yellow', 'orange', 'red', 'purple']
        dist.plot(kind='bar', ax=axes[0], color=colors[:len(dist)])
        axes[0].set_title('Risk Level Distribution')
        axes[0].set_xlabel('Risk Level')
        axes[0].set_ylabel('Count')
        
        # Pie chart
        dist.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', colors=colors[:len(dist)])
        axes[1].set_title('Risk Level Proportions')
        axes[1].set_ylabel('')
        
        plt.tight_layout()
        plt.savefig('../eda_results/risk_distribution.png')
        plt.show()
        
        self.results['distribution'] = {
            'counts': dist.to_dict(),
            'percentages': dist_pct.to_dict(),
            'imbalance_ratio': imbalance_ratio
        }
    
    def analyze_feature_separability(self):
        """
        Check if features can separate risk levels
        """
        print("\n" + "="*60)
        print("📊 FEATURE SEPARABILITY ANALYSIS")
        print("="*60)
        
        # Encode risk levels for analysis
        risk_order = {'Green':0, 'Yellow':1, 'Orange':2, 'Red':3, 'Purple':4}
        self.df['risk_encoded'] = self.df[self.target_col].map(risk_order)
        
        # Analyze each feature
        print("\n📈 Risk Level Statistics by Feature:")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(self.feature_cols[:5]):  # First 5 features
            ax = axes[idx]
            
            # Box plot
            self.df.boxplot(column=feature, by=self.target_col, ax=ax)
            ax.set_title(f'{feature} by Risk Level')
            ax.set_xlabel('Risk Level')
            
            # Calculate separability metrics
            risk_stats = self.df.groupby(self.target_col)[feature].agg(['mean', 'std']).round(2)
            print(f"\n  {feature}:")
            for risk in risk_stats.index:
                print(f"    {risk}: mean={risk_stats.loc[risk, 'mean']:.1f}, std={risk_stats.loc[risk, 'std']:.1f}")
            
            # ANOVA F-test for separability
            from scipy import stats
            risk_groups = [self.df[self.df[self.target_col] == risk][feature].dropna() 
                          for risk in self.df[self.target_col].unique()]
            f_stat, p_value = stats.f_oneway(*risk_groups)
            
            print(f"    ANOVA F-test: F={f_stat:.2f}, p={p_value:.4f}")
            if p_value < 0.05:
                print(f"    ✅ {feature} significantly differs across risk levels")
            else:
                print(f"    ❌ {feature} does NOT significantly differ across risk levels")
        
        plt.suptitle('Feature Distributions by Risk Level')
        plt.tight_layout()
        plt.savefig('../eda_results/feature_separability.png')
        plt.show()
    
    def analyze_consistency(self):
        """
        Check if similar weather gets same risk level
        """
        print("\n" + "="*60)
        print("📊 CONSISTENCY ANALYSIS")
        print("="*60)
        
        # Find similar weather patterns
        print("\n🔍 Checking for inconsistent risk labels...")
        
        # Use a sample for performance
        sample_df = self.df.sample(min(1000, len(self.df)))
        inconsistent_cases = []
        
        for idx, row in sample_df.iterrows():
            # Find similar weather (within thresholds)
            similar = self.df[
                (abs(self.df['temperature'] - row['temperature']) < 2) &
                (abs(self.df['wind_speed'] - row['wind_speed']) < 10) &
                (abs(self.df['precipitation'] - row['precipitation']) < 5)
            ]
            
            if len(similar) >= 5:  # At least 5 similar cases
                unique_risks = similar['risk_level'].nunique()
                if unique_risks >= 3:  # Found 3+ different risk levels for similar weather
                    inconsistent_cases.append({
                        'weather': {
                            'temp': row['temperature'],
                            'wind': row['wind_speed'],
                            'rain': row['precipitation']
                        },
                        'risks_found': similar['risk_level'].unique().tolist(),
                        'count': len(similar)
                    })
        
        print(f"\n📌 Found {len(inconsistent_cases)} inconsistent weather-risk mappings")
        
        if inconsistent_cases:
            print("\n📋 Examples of inconsistent labeling:")
            for i, case in enumerate(inconsistent_cases[:5]):
                print(f"\n  Case {i+1}:")
                print(f"    Weather: T={case['weather']['temp']:.1f}°C, "
                      f"W={case['weather']['wind']:.1f}km/h, "
                      f"R={case['weather']['rain']:.1f}mm")
                print(f"    Different risk levels assigned: {', '.join(case['risks_found'])}")
                print(f"    Number of similar cases: {case['count']}")
        
        # Calculate consistency score
        consistency_score = max(0, 100 - len(inconsistent_cases) * 2)
        print(f"\n📊 Consistency Score: {consistency_score:.1f}/100")
        
        self.results['consistency'] = {
            'inconsistent_count': len(inconsistent_cases),
            'consistency_score': consistency_score,
            'examples': inconsistent_cases[:5]
        }
    
    def analyze_cluster_agreement(self):
        """
        Compare existing labels with natural clusters
        """
        print("\n" + "="*60)
        print("📊 CLUSTER AGREEMENT ANALYSIS")
        print("="*60)
        
        # Prepare data
        X = self.df[self.feature_cols].copy()
        X_scaled = StandardScaler().fit_transform(X)
        
        # Encode existing labels
        risk_order = {'Green':0, 'Yellow':1, 'Orange':2, 'Red':3, 'Purple':4}
        y_true = self.df[self.target_col].map(risk_order).values
        
        # Try different clustering algorithms
        clustering_algorithms = {
            'KMeans (5 clusters)': KMeans(n_clusters=5, random_state=42),
            'Agglomerative (5 clusters)': AgglomerativeClustering(n_clusters=5),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=10)
        }
        
        results = {}
        
        for name, algorithm in clustering_algorithms.items():
            print(f"\n📌 {name}:")
            
            # Fit and predict
            labels = algorithm.fit_predict(X_scaled)
            
            # Handle DBSCAN noise
            if name == 'DBSCAN':
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                print(f"  Clusters found: {n_clusters}")
                print(f"  Noise points: {n_noise}")
                
                # Filter out noise for metrics
                mask = labels != -1
                if mask.sum() > 0:
                    ari = adjusted_rand_score(y_true[mask], labels[mask])
                    sil = silhouette_score(X_scaled[mask], labels[mask])
                else:
                    ari = 0
                    sil = 0
            else:
                n_clusters = len(set(labels))
                ari = adjusted_rand_score(y_true, labels)
                sil = silhouette_score(X_scaled, labels)
            
            print(f"  Adjusted Rand Index: {ari:.3f}")
            print(f"  Silhouette Score: {sil:.3f}")
            
            # Interpret ARI
            if ari > 0.5:
                print("  ✅ Strong agreement with natural clusters")
            elif ari > 0.3:
                print("  ⚠️ Moderate agreement with natural clusters")
            elif ari > 0.1:
                print("  ❌ Weak agreement with natural clusters")
            else:
                print("  ❌ No agreement with natural clusters")
            
            results[name] = {
                'ari': ari,
                'silhouette': sil,
                'n_clusters': n_clusters
            }
        
        self.results['cluster_agreement'] = results
        
        # Visualize best clustering
        best_method = max(results.items(), key=lambda x: x[1]['ari'])
        print(f"\n🏆 Best clustering method: {best_method[0]}")
        
        # Get labels from best method
        if best_method[0] == 'DBSCAN':
            best_labels = DBSCAN(eps=0.5, min_samples=10).fit_predict(X_scaled)
        elif best_method[0] == 'Agglomerative (5 clusters)':
            best_labels = AgglomerativeClustering(n_clusters=5).fit_predict(X_scaled)
        else:
            best_labels = KMeans(n_clusters=5, random_state=42).fit_predict(X_scaled)
        
        # Visualize
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Existing labels
        scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', alpha=0.6)
        axes[0].set_title('Existing Risk Labels')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Discovered clusters
        scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels, cmap='viridis', alpha=0.6)
        axes[1].set_title(f'Natural Clusters ({best_method[0]})')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('../eda_results/cluster_comparison.png')
        plt.show()
    
    def analyze_model_baseline(self):
        """
        Train simple models to establish baseline
        """
        print("\n" + "="*60)
        print("📊 MODEL BASELINE PERFORMANCE")
        print("="*60)
        
        # Prepare data
        X = self.df[self.feature_cols].copy()
        risk_order = {'Green':0, 'Yellow':1, 'Orange':2, 'Red':3, 'Purple':4}
        y = self.df[self.target_col].map(risk_order)
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test different models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        }
        
        results = {}
        
        print("\n📈 Model Performance:")
        for name, model in models.items():
            print(f"\n  {name}:")
            
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            print(f"    Train Accuracy: {train_score:.3f}")
            print(f"    Test Accuracy:  {test_score:.3f}")
            print(f"    CV Accuracy:    {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(self.feature_cols, model.feature_importances_))
                print(f"    Top features: {sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
            
            results[name] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n🏆 Best Model: {best_model[0]} with {best_model[1]['test_accuracy']:.3f} test accuracy")
        
        # Determine if target is learnable
        if best_model[1]['test_accuracy'] > 0.7:
            print("✅ Target is learnable! Models perform well.")
        elif best_model[1]['test_accuracy'] > 0.5:
            print("⚠️ Target is somewhat learnable, but room for improvement.")
        else:
            print("❌ Target is difficult to learn. Consider redefining.")
        
        self.results['model_baseline'] = results
        self.results['best_model_accuracy'] = best_model[1]['test_accuracy']
    
    def generate_recommendation(self):
        """
        Generate final recommendation based on all analyses
        """
        print("\n" + "="*60)
        print("🎯 FINAL RECOMMENDATION")
        print("="*60)
        
        # Calculate composite score
        scores = []
        
        # Distribution balance
        imbalance = self.results['distribution']['imbalance_ratio']
        if imbalance < 3:
            scores.append(100)
        elif imbalance < 6:
            scores.append(70)
        elif imbalance < 10:
            scores.append(40)
        else:
            scores.append(20)
        
        # Consistency
        scores.append(self.results['consistency']['consistency_score'])
        
        # Cluster agreement (best ARI)
        best_ari = max([v['ari'] for v in self.results['cluster_agreement'].values()])
        scores.append(best_ari * 100)
        
        # Model performance
        scores.append(self.results['best_model_accuracy'] * 100)
        
        # Average score
        final_score = np.mean(scores)
        
        print(f"\n📊 Target Variable Quality Score: {final_score:.1f}/100")
        
        print("\n📋 Detailed Scores:")
        print(f"  • Distribution Balance: {scores[0]:.1f}/100")
        print(f"  • Label Consistency: {scores[1]:.1f}/100")
        print(f"  • Natural Cluster Agreement: {scores[2]:.1f}/100")
        print(f"  • Model Learnability: {scores[3]:.1f}/100")
        
        # Final recommendation
        print("\n🔍 ANALYSIS SUMMARY:")
        
        if final_score >= 80:
            print("✅ EXCELLENT: Your target variable is well-defined!")
            print("\n📌 RECOMMENDATION: Proceed with supervised learning")
            print("   • Use XGBoost/Random Forest as planned")
            print("   • Focus on feature engineering")
            print("   • Add more historical data for better performance")
            
            self.results['recommendation'] = 'supervised'
            
        elif final_score >= 60:
            print("⚠️ GOOD: Your target variable is decent but could be improved")
            print("\n📌 RECOMMENDATION: Use hybrid approach")
            print("   • Use clustering insights to refine labels")
            print("   • Check inconsistent cases and relabel if needed")
            print("   • Consider using soft labels or probability targets")
            
            self.results['recommendation'] = 'hybrid'
            
        elif final_score >= 40:
            print("⚠️ WEAK: Your target variable needs significant improvement")
            print("\n📌 RECOMMENDATION: Refine target definition")
            print("   • Review your risk labeling criteria")
            print("   • Use clustering to discover natural risk patterns")
            print("   • Consider multi-task learning for different danger types")
            
            self.results['recommendation'] = 'refine'
            
        else:
            print("❌ POOR: Your target variable is not well-defined")
            print("\n📌 RECOMMENDATION: Start fresh with unsupervised learning")
            print("   • Use clustering to discover natural risk groups")
            print("   • Define new risk levels based on cluster profiles")
            print("   • Build labeled dataset from cluster assignments")
            
            self.results['recommendation'] = 'unsupervised'
        
        # Save results
        self.results['final_score'] = final_score
        self.results['analysis_date'] = datetime.now().isoformat()
        
        # Save to file
        with open('../eda_results/target_analysis_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: ../eda_results/target_analysis_results.json")
        
        return self.results
    
    def run_full_analysis(self):
        """
        Run all analyses
        """
        print("\n" + "="*60)
        print("🚀 STARTING COMPREHENSIVE TARGET VARIABLE ANALYSIS")
        print("="*60)
        
        # Load data
        if not self.load_and_prepare_data():
            return None
        
        # Run analyses
        self.analyze_distribution()
        self.analyze_feature_separability()
        self.analyze_consistency()
        self.analyze_cluster_agreement()
        self.analyze_model_baseline()
        
        # Generate recommendation
        results = self.generate_recommendation()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print("="*60)
        
        return results

def main():
    """
    Main function to run the analysis
    """
    # Create output directory if it doesn't exist
    os.makedirs('../eda_results', exist_ok=True)
    
    # Initialize analyzer
    analyzer = TargetVariableAnalyzer()
    
    # Run analysis
    results = analyzer.run_full_analysis()
    
    if results:
        print(f"\n📊 Final Score: {results['final_score']:.1f}/100")
        print(f"🎯 Recommendation: {results['recommendation'].upper()}")
        
        # Provide next steps based on recommendation
        print("\n📋 NEXT STEPS:")
        
        if results['recommendation'] == 'supervised':
            print("""
            1. Continue with your XGBoost/Random Forest models
            2. Focus on feature engineering:
               - Add temporal features (hour, day, month)
               - Add lag features (previous day weather)
               - Add governorate-specific features
            3. Use cross-validation for robust evaluation
            4. Consider ensemble methods for better performance
            """)
            
        elif results['recommendation'] == 'hybrid':
            print("""
            1. Use clustering insights to refine labels
            2. Create a refined target variable:
               - Keep labels that match clusters
               - Adjust labels that are inconsistent
               - Consider adding a "uncertain" category
            3. Retrain models on refined labels
            4. Compare performance with original
            """)
            
        elif results['recommendation'] == 'refine':
            print("""
            1. Review your risk labeling criteria with domain experts
            2. Consider objective thresholds based on:
               - Historical damage data
               - Official warnings
               - Impact on different user groups
            3. Use clustering to identify natural risk levels
            4. Create new labels based on findings
            """)
            
        else:  # unsupervised
            print("""
            1. Switch to unsupervised learning approach
            2. Let clusters define risk levels:
               - Run KMeans with 3-7 clusters
               - Analyze cluster profiles
               - Label clusters based on severity
            3. Use cluster assignments as new target
            4. Build supervised model on new labels
            """)

if __name__ == "__main__":
    main()