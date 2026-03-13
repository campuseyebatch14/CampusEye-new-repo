from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
uri = os.getenv('MONGODB_URI')
print("URI:", uri)  # Debug
if not uri or not uri.startswith(('mongodb://', 'mongodb+srv://')):
    raise ValueError("MONGODB_URI is invalid or not set. Check .env file.")
client = MongoClient(uri)
db = client['student_surveillance']
students_collection = db['students']
detections_collection = db['detections']

DISTANCE_THRESHOLD = 10

def deleteStudent(student_id):
    students_collection.delete_one({'studentId': student_id})

def getStudentDetails(student_id):
    query = students_collection.find_one(
        {'studentId': student_id},
        {
            'name': 1,
            'studentId': 1,
            'branch': 1,
            'photoUrl': 1,
            '_id': 0
        }
    )
    return query

def getSuspectsDetails(suspect_ids):
    query = students_collection.find(
        {'studentId': {"$in": suspect_ids}},
        {
            'name': 1,
            'studentId': 1,
            'branch': 1,
            'photoUrl': 1,
            '_id': 0
        }
    )
    return list(query)

def findMatch(target_embedding):
    query = students_collection.aggregate([
        {
            "$addFields": {
                "target_embedding": target_embedding
            }
        }, {"$unwind": {"path": "$embedding", "includeArrayIndex": "embedding_index"}},
        {"$unwind": {"path": "$target_embedding", "includeArrayIndex": "target_index"}},
        {
            "$project": {
                "studentId": 1,
                "embedding": 1,
                "target_embedding": 1,
                "compare": {
                    "$cmp": ['$embedding_index', '$target_index']
                }
            }
        }, {"$match": {"compare": 0}},
        {
            "$group": {
                "_id": "$studentId",
                "distance": {
                    "$sum": {
                        "$pow": [{
                            "$subtract": ['$embedding', '$target_embedding']
                        }, 2]
                    }
                }
            }
        },
        {
            "$project": {
                "_id": 1,
                "distance": {"$sqrt": "$distance"}
            }
        },
        {
            "$project": {
                "_id": 1, "distance": 1, "cond": {"$lte": ["$distance", DISTANCE_THRESHOLD]}
            }
        }, {"$match": {"cond": True}}, {"$sort": {"distance": 1}}, {"$limit": DISTANCE_THRESHOLD}
    ])

    return list(query)

def store_detection_records(records):
    detections_collection.insert_many(records)