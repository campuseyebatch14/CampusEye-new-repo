import os
import gc
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# --- CONNECTION LOGIC ---
# We check BOTH 'MONGODB_URI' and 'MONGO_URI' to be safe
uri = os.getenv('MONGODB_URI') or os.getenv('MONGO_URI')

# If the URI is missing, we stop here and show a clear error in Render Logs
if not uri:
    print("CRITICAL ERROR: No MongoDB URI found. Check Render Environment Settings.")
    raise ValueError("MONGODB_URI is missing. Please add it to Render -> Settings -> Environment.")

try:
    # serverSelectionTimeoutMS=10000 gives the database 10 seconds to connect
    # tlsAllowInvalidCertificates=True helps if there are SSL/Certificate issues on Render
    client = MongoClient(uri, serverSelectionTimeoutMS=10000, tlsAllowInvalidCertificates=True)
    
    # Change 'student_surveillance' to your actual Database name if it's different in Atlas
    db = client['student_surveillance']
    students_collection = db['students']
    detections_collection = db['detections']
    
    # Quick Test: Verify connection immediately
    client.admin.command('ping')
    print("MongoDB Connected Successfully!")
except Exception as e:
    print(f"MongoDB Connection Failed: {str(e)}")
    raise

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
    """Fetches details for multiple students at once."""
    return list(students_collection.find(
        {'studentId': {"$in": suspect_ids}},
        {'name': 1, 'studentId': 1, 'branch': 1, 'photoUrl': 1, '_id': 0}
    ))

def store_detection_records(records):
    """Logs identified students into the detections collection."""
    if records:
        detections_collection.insert_many(records)
