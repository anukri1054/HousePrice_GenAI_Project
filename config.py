import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")

# Feature names path
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")

# Output directory
OUTPUT_DIR = BASE_DIR
