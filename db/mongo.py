from pymongo import MongoClient

def get_db():
    db_client = MongoClient("mongodb://localhost:27017/")
    return db_client["text_analysis_db"]