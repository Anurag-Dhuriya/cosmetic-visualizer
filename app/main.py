# app/main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.routes import upload, analysis, transform, save  # ADD transform and save
from app.config import UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR

# Create FastAPI application
app = FastAPI(
    title="Cosmetic Treatment Visualizer API",
    description="Backend API for cosmetic treatment visualization",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router)
app.include_router(analysis.router)
app.include_router(transform.router)  # ADD THIS
app.include_router(save.router)       # ADD THIS

# Serve static files
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/temp", StaticFiles(directory=str(TEMP_DIR)), name="temp")

@app.get("/")
async def root():
    return {
        "message": "Cosmetic Treatment Visualizer API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload": "/api/upload/image",
            "analyze": "/api/analysis/detect-face",
            "transform": "/api/transform/apply",
            "save": "/api/save/result",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}