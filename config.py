import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

RANDOM_STATE = 42

TEST_SIZE = 0.20