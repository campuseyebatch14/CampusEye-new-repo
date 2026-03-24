import os
import gc
import numpy as np
from deepface import DeepFace
import mongo_utils

# --- LIGHTWEIGHT SETTINGS ---
# Using OpenFace as it's ~4x smaller than VGG-Face
# Using opencv detector as it's the fastest and uses the least RAM
MODEL_NAME = 'OpenFace'
DETECTOR = 'opencv'

def getEmbedding(img):
    """Generates a face embedding while keeping RAM usage minimal."""
    try:
        # We use silent=True to prevent heavy logging overhead
        results = DeepFace.represent(
            img_path=img, 
            model_name=MODEL_NAME, 
            detector_backend=DETECTOR, 
            enforce_detection=True
        )
        
        if results and len(results) > 0:
            embedding = results[0]['embedding']
            return embedding
        return None
    except Exception as e:
        # If no face is detected or error occurs, return None
        return None

def findSuspects(frame):
    """Identifies students by comparing live embeddings with the database."""
    found_ids = []
    try:
        # 1. Generate embedding for the current browser frame
        target_embedding = getEmbedding(frame)
        
        if target_embedding is None:
            return {'found_suspect_ids': []}

        target_emb_np = np.array(target_embedding)

        # 2. Fetch student embeddings from MongoDB
        # We only fetch studentId and embedding to keep the data transfer light
        students = list(mongo_utils.students_collection.find(
            {}, 
            {'studentId': 1, 'embedding': 1, '_id': 0}
        ))
        
        for student in students:
            db_emb_np = np.array(student['embedding'])
            
            # 3. Manual Cosine Similarity calculation
            # Formula: (A . B) / (||A|| * ||B||)
            dot_product = np.dot(target_emb_np, db_emb_np)
            norm_a = np.linalg.norm(target_emb_np)
            norm_b = np.linalg.norm(db_emb_np)
            
            similarity = dot_product / (norm_a * norm_b)
            
            # 0.85 similarity is a strong match (roughly 14-15 distance)
            if similarity > 0.85:
                found_ids.append(student['studentId'])

        # 4. --- CRITICAL MEMORY CLEANUP ---
        # Explicitly delete large arrays to free up 512MB limit
        del target_emb_np
        del students
        gc.collect() 
                
        return {'found_suspect_ids': found_ids}
        
    except Exception as e:
        print(f"AI Model Error: {e}")
        return {'found_suspect_ids': []}
