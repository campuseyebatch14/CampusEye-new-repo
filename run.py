from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
uri = os.getenv('MONGODB_URI')
print("URI:", uri)  # Debug print
if not uri or not uri.startswith(('mongodb://', 'mongodb+srv://')):
    raise ValueError("MONGODB_URI is invalid or not set. Check .env file.")
client = MongoClient(uri)
db = client['student_surveillance']
print("Databases:", client.list_database_names())
client.close()