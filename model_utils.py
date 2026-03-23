from deepface import DeepFace
import numpy as np
import mongo_utils

def getEmbedding(img):
    try:
        res = DeepFace.represent(img, model_name='OpenFace', detector_backend='opencv', enforce_detection=True)
        return res[0]['embedding']
    except: return None

def findSuspects(frame):
    found_ids = []
    try:
        # 1. Get embedding of live frame
        target_embedding = getEmbedding(frame)
        if target_embedding is None: return {'found_suspect_ids': []}

        # 2. Compare against MongoDB embeddings
        students = list(mongo_utils.students_collection.find({}, {'studentId': 1, 'embedding': 1}))
        
        for student in students:
            db_embedding = np.array(student['embedding'])
            # Manual Cosine Similarity to save RAM
            dist = np.dot(target_embedding, db_embedding) / (np.linalg.norm(target_embedding) * np.linalg.norm(db_embedding))
            # 0.85 similarity = ~14-15 distance threshold
            if dist > 0.85: 
                found_ids.append(student['studentId'])
                
        return {'found_suspect_ids': found_ids}
    except: return {'found_suspect_ids': []}
