# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from .api.routes import router

# app = FastAPI(
#     title="WeatherGuardTN API",
#     description="Early weather danger & vigilance prediction for Tunisia",
#     version="1.0.0"
# )

# # Allow React frontend (Vite default port 5173)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173", "*"],  
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.include_router(router)

# @app.get("/")
# def root():
#     return {"message": "WeatherGuardTN API is running"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import mlflow
import os
from .api.routes import router

# Configuration MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Configuration de l'expérience MLflow
EXPERIMENT_NAME = "WeatherGuardTN"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestion du cycle de vie de l'application
    - Au démarrage : configure MLflow
    - À l'arrêt : cleanup si nécessaire
    """
    # --- STARTUP ---
    print(f"🚀 Démarrage de WeatherGuardTN API...")
    print(f"📊 MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    
    # Créer ou obtenir l'expérience MLflow
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"✅ Nouvelle expérience MLflow créée: {EXPERIMENT_NAME}")
    else:
        experiment_id = experiment.experiment_id
        print(f"✅ Expérience MLflow existante: {EXPERIMENT_NAME}")
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Test de connexion à MLflow
    try:
        mlflow.search_experiments()
        print("✅ Connexion à MLflow établie")
    except Exception as e:
        print(f"⚠️  Attention: Impossible de se connecter à MLflow: {e}")
    
    yield
    
    # --- SHUTDOWN ---
    print("👋 Arrêt de WeatherGuardTN API...")

# Initialisation de l'application avec lifespan
app = FastAPI(
    title="WeatherGuardTN API",
    description="Early weather danger & vigilance prediction for Tunisia",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc UI
    openapi_url="/openapi.json"  # Schéma OpenAPI
)

# Configuration CORS pour les environnements Docker et local
origins = [
    "http://localhost:3000",  # Frontend Docker
    "http://localhost:5173",  # Frontend Vite (dev)
    "http://localhost:80",    # Frontend Nginx
    "http://frontend:80",     # Frontend dans Docker network
    "*"  # Pour le développement (à restreindre en production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes
app.include_router(router, prefix="/api")

@app.get("/")
def root():
    return {
        "message": "WeatherGuardTN API is running",
        "version": "1.0.0",
        "status": "healthy",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
def health_check():
    """
    Endpoint pour les health checks
    """
    return {
        "status": "healthy",
        "mlflow_connected": test_mlflow_connection()
    }

def test_mlflow_connection():
    """
    Teste la connexion à MLflow
    """
    try:
        mlflow.search_experiments()
        return True
    except:
        return False

@app.get("/mlflow/info")
def mlflow_info():
    """
    Informations sur la configuration MLflow
    """
    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "experiment": EXPERIMENT_NAME,
        "status": "connected" if test_mlflow_connection() else "disconnected"
    }