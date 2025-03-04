from pymongo import MongoClient
import os

def get_db():
    db_client = MongoClient(os.getenv("MONGO_URI"))
    return db_client[os.getenv("MONGO_DB")]