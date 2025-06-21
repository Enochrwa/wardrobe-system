from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import os
import logging
import tensorflow as tf
from app.db import database
from app.db.database import Base
from app import models
from app.routers import (
    auth,
    wardrobe,
    outfits,
    weekly_plans,
    occasions,
    style_history,
    statistics,
    ai_analyzer,
    recommendations,
    user_profile,
    community,
    model_cache,  # Added model cache router
    ml_features,  # Added ML features router
    enhanced_search,  # Added enhanced search router
    occasion_recommendations,  # Added occasion recommendations router
)

# Disable oneDNN optimizations if desired
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Use FastAPI lifespan event to manage app startup and shutdown.
    Execute DDL statements in development mode only.
    Preload AI models for better performance.
    """
    if os.getenv("ENV") == "development" and os.getenv("RUN_MAIN") == "true":
        Base.metadata.drop_all(bind=database.engine)
        Base.metadata.create_all(bind=database.engine)
    
    # Preload AI models on startup
    try:
        from app.services.model_manager import get_model_manager
        model_manager = get_model_manager()
        logging.info("Preloading AI models...")
        
        # Preload models in background
        for model_name in ["mobilenet_v2", "efficientdet_lite0"]:
            try:
                model = model_manager.get_model(model_name)
                if model is not None:
                    logging.info(f"Model {model_name} preloaded successfully")
                else:
                    logging.warning(f"Failed to preload model {model_name}")
            except Exception as e:
                logging.error(f"Error preloading model {model_name}: {e}")
        
        logging.info("AI models preloading completed")
    except Exception as e:
        logging.error(f"Error during model preloading: {e}")
    
    yield

app = FastAPI(lifespan=lifespan)

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://localhost:5713",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5713",
    "https://digital-wardrobe-system.vercel.app",
    "*",  # Allow all origins for development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# API routers
app.include_router(auth.router, prefix="/api")
app.include_router(wardrobe.router, prefix="/api")
app.include_router(outfits.router, prefix="/api")
app.include_router(weekly_plans.router, prefix="/api")
app.include_router(occasions.router, prefix="/api")
app.include_router(style_history.router, prefix="/api")
app.include_router(statistics.router, prefix="/api")
app.include_router(ai_analyzer.router, prefix="/api")
app.include_router(recommendations.router, prefix="/api")
app.include_router(user_profile.router, prefix="/api")
app.include_router(community.router, prefix="/api")
app.include_router(model_cache.router, prefix="/api")  # Added model cache router
app.include_router(ml_features.router, prefix="/api")  # Added ML features router
app.include_router(enhanced_search.router, prefix="/api")  # Added enhanced search router
app.include_router(occasion_recommendations.router, prefix="/api")  # Added occasion recommendations router

@app.get("/")
async def root():
    return {"message": "Digital Wardrobe System API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring"""
    return {"status": "healthy", "service": "digital-wardrobe-backend"}

# Entry point
if __name__ == "__main__":
    import uvicorn
    # In development, reload=True will set RUN_MAIN for our startup guard
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
    )
