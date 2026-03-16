import os
import sys

# --- STEP 1: THE CRITICAL DEPLOYMENT HACK ---
# This MUST happen before any DeepFace or TensorFlow imports.
# It tricks DeepFace into finding the Keras modules it expects.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    import tensorflow as tf
    import keras
    # Manually inject keras into the tensorflow namespace
    sys.modules["tensorflow.keras"] = keras
    sys.modules["tensorflow.keras.preprocessing"] = keras.preprocessing
except ImportError:
    print("TensorFlow or Keras not found, deployment might fail.")

# --- STEP 2: STANDARD IMPORTS ---
import cv2
import numpy as np
from deepface import DeepFace
import mongo_utils

# Use Facenet; it is significantly lighter than VGG-Face for Render's 512MB RAM
MODEL = 'Facenet' 
DETECTOR = 'opencv' 

def getRepresentations(img):
    """Generates face representations from a BGR numpy array."""
    try:
        # enforce_detection=False prevents the code from crashing if no face is seen
        obj = DeepFace.represent(
            img_path=img,
            model_name=MODEL,
            detector_backend=DETECTOR,
            enforce_detection=False 
        )
        return obj
    except Exception as e:
        print(f'No face detected or model error: {str(e)}')
        return None

def getEmbedding(img):
    """Extracts the first embedding found in the image."""
    try:
        obj = getRepresentations(img)
        if not obj or len(obj) == 0:
            return None
        return obj[0]['embedding']
    except Exception as e:
        print(f'Error generating embedding: {str(e)}')
        return None

def drawRectangle(img, facial_area):
    """Draws a cyan bounding box around detected faces."""
    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
    return cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

def findSuspects(input_img):
    """Matches detected faces against the MongoDB database."""
    try:
        input_representations = getRepresentations(input_img)
        
        if not input_representations:
            return {'found_suspect_ids': [], 'suspects_img': input_img}
            
        found_suspect_ids = []
        matched_rep_ids = []
        
        for index, rep in enumerate(input_representations):
            # Only process if an embedding was actually generated
            if 'embedding' in rep:
                res = mongo_utils.findMatch(rep['embedding'])
                if res and len(res) > 0:
                    matched_rep_ids.append(index)
                    found_suspect_ids.append(res[0]['_id'])
    
        # Draw boxes only for matched suspects
        suspects_img = input_img.copy()
        for idx in matched_rep_ids:
            facial_area = input_representations[idx]['facial_area']
            suspects_img = drawRectangle(suspects_img, facial_area)
              
        return {'found_suspect_ids': found_suspect_ids, 'suspects_img': suspects_img}
    
    except Exception as e:
        print(f'Error in findSuspects: {str(e)}')
        return {'found_suspect_ids': [], 'suspects_img': input_img}
