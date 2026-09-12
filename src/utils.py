"""
Shared constants and helper functions used across
train.py, evaluate.py, predict.py, and app.py
"""

import os
import numpy as np
from PIL import Image

# ----------------------------
# Constants
# ----------------------------
IMG_SIZE = 128                 # width/height images are resized to
CHANNELS = 3
BATCH_SIZE = 32
CLASS_NAMES = ["apple", "banana", "orange", "mango", "grapes"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fruit_classifier_model.h5")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CLASS_INDICES_PATH = os.path.join(OUTPUT_DIR, "class_indices.json")


def ensure_dirs():
    """Create model/output directories if they don't exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_image(image, img_size=IMG_SIZE):
    """
    Preprocess a single image for prediction.
    Accepts a PIL.Image, a file path (str), or a numpy array (OpenCV/BGR).
    Returns a normalized numpy array of shape (1, img_size, img_size, 3).
    """
    if isinstance(image, str):
        # Load from file path with OpenCV, convert BGR -> RGB
       from PIL import Image
       import numpy as np

       img = Image.open(image).convert("RGB")   
    elif isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        img = image
        if img.shape[-1] == 4:  # RGBA -> RGB
            img = img[..., :3]
    else:
        raise TypeError("Unsupported image type for preprocessing.")

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # add batch dimension
    return img