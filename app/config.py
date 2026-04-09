# app/config.py

import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory paths
UPLOAD_DIR = BASE_DIR / "uploads"
TEMP_DIR = BASE_DIR / "temp"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# File upload settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Image processing settings
MAX_IMAGE_DIMENSION = 2048  # Maximum width or height

# Treatment categories
TREATMENT_CATEGORIES = {
    "face": [
        "temples_fillers",
        "cheek_fillers",
        "chin_fillers",
        "jawline_contouring",
        "forehead_lines",
        "glabellar_lines",
        "nasolabial_folds",
        "marionette_folds"
    ],
    "lips": [
        "plumper",
        "cupids_bow",
        "upper_lip_fillers",
        "lower_lip_fillers",
        "corner_lip_lift_fillers"
    ],
    "nose": [
        "contouring",
        "bridge_fillers",
        "root_fillers",
        "tip_lift_fillers",
        "slimming_fillers"
    ],
    "eyebrows": [
        "brow_lift"
    ]
}