# app/main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.routes import upload, analysis, transform
from app.config import UPLOAD_DIR, TEMP_DIR

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
app.include_router(transform.router)

# Serve static files
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
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
            "transform_multiple": "/api/transform/apply-multiple",
            "categories": "/api/transform/categories",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}