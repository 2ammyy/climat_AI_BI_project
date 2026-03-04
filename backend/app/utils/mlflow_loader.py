# backend/app/utils/mlflow_loader.py - Update to load refined model

import mlflow
import joblib
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_path=None, use_refined=True):
        self.model_path = model_path or "../mlartifacts"
        self.use_refined = use_refined
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.feature_names = ['temperature', 'wind_speed', 'precipitation', 'humidity', 'pressure']
        
    def load_model(self):
        """Load the best model"""
        try:
            if self.use_refined:
                # Load refined model
                model_file = Path(self.model_path) / "refined_risk_model.pkl"
                encoder_file = Path(self.model_path) / "label_encoder.pkl"
                scaler_file = Path(self.model_path) / "scaler.pkl"
                
                if model_file.exists():
                    self.model = joblib.load(model_file)
                    self.label_encoder = joblib.load(encoder_file)
                    self.scaler = joblib.load(scaler_file)
                    logger.info(f"✅ Loaded refined model from {model_file}")
                else:
                    logger.warning("Refined model not found, falling back to original")
                    self._load_original_model()
            else:
                self._load_original_model()
                
            return self.model is not None
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _load_original_model(self):
        """Load original model from MLflow"""
        try:
            import glob
            # Find most recent model
            model_files = glob.glob(f"{self.model_path}/**/model.pkl", recursive=True)
            if model_files:
                latest_model = max(model_files, key=os.path.getmtime)
                self.model = joblib.load(latest_model)
                logger.info(f"✅ Loaded original model from {latest_model}")
            else:
                logger.error("No model files found")
        except Exception as e:
            logger.error(f"Error loading original model: {e}")
    
    def predict(self, features):
        """Make prediction"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Scale features if scaler exists
        if self.scaler:
            features_scaled = self.scaler.transform([features])
        else:
            features_scaled = [features]
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        # Decode if label encoder exists
        if self.label_encoder:
            prediction = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            return prediction, probabilities
        else:
            return prediction, None

# Singleton instance
_model_loader = None

def get_model_loader():
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader(use_refined=True)
        _model_loader.load_model()
    return _model_loader