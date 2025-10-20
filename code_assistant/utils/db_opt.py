import sys
from dotenv import load_dotenv
from pymongo import MongoClient

def get_mongo_client(mongo_uri):
    if not mongo_uri:
        print("Error: MONGO_URI environment variable is not set in .env")
        sys.exit(1)

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=30000)
    try:
        client.admin.command('ping')
        print("MongoDB connection successful.")
    except Exception as e:
        print(f"MongoDB connection error: {str(e)}")
        sys.exit(1) 
    return client