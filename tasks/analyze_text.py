import re
from datetime import datetime

from bson import ObjectId
from celery import Celery
import logging
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter
from pymongo import MongoClient

from tasks.analyze_custom_model import analyze_text as analyze_custom_model
from string import punctuation
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)


def get_db():
    db_client = MongoClient(os.getenv("MONGO_URI"))
    return db_client[os.getenv("MONGO_DB")]


app = Celery('analyze_tasks', backend=os.getenv("REDIS_URI"), broker=os.getenv("REDIS_URI"))

@app.task
def analyze_text(text, job_id):
    analysis_collection, sentiment_collection, job_collection, sentences = prepare_analyzing(job_id, text)
    result = []
    tagger = Classifier.load('ner')
    tagger.predict(sentences, mini_batch_size=64)
    for sentence in sentences:
        for entity in sentence.get_labels():
            result.append({
                "type": entity.value,
                "score": entity.score,
                "text": entity.data_point.text
            })

    document = {
        "job_id": ObjectId(job_id),
        "model": "Flair",
        "text": text,
        "analysis": result
    }
    analysis_collection.insert_one(document)

    job_collection.update_one({"_id": ObjectId(job_id)},
                             {"$set": {"status_ner": "Completed", "end_time": datetime.now()}})

    logger.info(f"Analysis of text '{text}' has been completed")


def prepare_analyzing(job_id, text):
    db = get_db()
    analysis_collection = db["analysis_results"]
    sentiment_collection = db["sentiment_results"]
    job_collection = db["jobs"]
    logger.info(f"job_id: {job_id}")
    job_collection.update_one({"_id": job_id}, {"$set": {"status_ner": "Running"}})

    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(text)
    return analysis_collection, sentiment_collection, job_collection, sentences


@app.task
def analyze_sentiment_text(text, job_id):
    analysis_collection, sentiment_collection, job_collection, sentences = prepare_analyzing(job_id, text)
    result = []
    sentiment = Classifier.load('sentiment')
    sentiment.predict(sentences, mini_batch_size=64)

    for sentence in sentences:
        for entity in sentence.get_labels():
            result.append({
                "sentiment": entity.value,
                "score": entity.score,
                "text": entity.data_point.text
            })

    document = {
        "job_id": ObjectId(job_id),
        "text": text,
        "sentiment": result
    }

    sentiment_collection.insert_one(document)

    job_collection.update_one({"_id": ObjectId(job_id)},
                             {"$set": {"status_sentiment": "Completed", "end_time": datetime.now()}})

    logger.info(f"Sentiment analysis of text '{text}' has been completed")


@app.task
def analyze_text_using_custom_model(text, job_id):
    analysis_collection, sentiment_collection, job_collection, sentences = prepare_analyzing(job_id, text)
    job_collection.update_one({"_id": job_id}, {"$set": {"status_ner": "Running"}})
    result = []
    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(text)
    for sentence in sentences:
        raw_result = analyze_custom_model(sentence.text)
        for word, label in raw_result:
            # Remove word starting with [ and ending with ] and remove punct
            if re.match(r"^\[.*\]$", word) or any(char in word for char in punctuation) or label == "O":
                continue
            result.append({
                "type": label.removeprefix("B-").removeprefix("I-"),
                "text": word
            })
    analysis_collection.insert_one({
        "job_id": ObjectId(job_id),
        "text": text,
        "model": "Custom Trained",
        "analysis": result
    })

    job_collection.update_one({"_id": ObjectId(job_id)},
                             {"$set": {"status_ner": "Completed", "end_time": datetime.now()}})
