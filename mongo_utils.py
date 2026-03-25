import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# --- CONNECTION LOGIC ---
# Standardizes on 'MONGODB_URI' for Render environment variables
uri = os.getenv('MONGODB_URI')

if not uri or not uri.startswith(('mongodb://', 'mongodb+srv://')):
    raise ValueError("MONGODB_URI is invalid or not set in Render Environment Settings.")

# Added serverSelectionTimeoutMS to prevent long-running 'DB Error' hangs
client = MongoClient(uri, serverSelectionTimeoutMS=5000)
db = client['student_surveillance']
students_collection = db['students']
detections_collection = db['detections']

# This threshold is used if you choose to use the MongoDB aggregation method
DISTANCE_THRESHOLD = 14

def deleteStudent(student_id):
    """Removes a student record by their unique ID."""
    students_collection.delete_one({'studentId': student_id})

def getStudentDetails(student_id):
    """Fetches public details for a single student."""
    return students_collection.find_one(
        {'studentId': student_id},
        {'name': 1, 'studentId': 1, 'branch': 1, 'photoUrl': 1, '_id': 0}
    )

def getSuspectsDetails(suspect_ids):
    """Fetches details for multiple students at once to minimize DB calls."""
    return list(students_collection.find(
        {'studentId': {"$in": suspect_ids}},
        {'name': 1, 'studentId': 1, 'branch': 1, 'photoUrl': 1, '_id': 0}
    ))

def store_detection_records(records):
    """Logs identified students into the detections collection."""
    if records:
        detections_collection.insert_many(records)
