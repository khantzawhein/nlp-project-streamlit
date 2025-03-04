import re
from datetime import datetime

from bson import ObjectId
from celery import Celery
import logging
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter
from db.mongo import get_db
from tasks.analyze_custom_model import analyze_text as analyze_custom_model
from string import punctuation

logger = logging.getLogger(__name__)

db = get_db()
analysis_collection = db["analysis_results"]
sentiment_collection = db["sentiment_results"]
jobCollection = db["jobs"]

app = Celery('analyze_tasks', backend="redis://localhost:6379/3", broker='redis://localhost:6379/3')


@app.task
def analyze_text(text, job_id):
    logger.info(f"job_id: {job_id}")
    jobCollection.update_one({"_id": job_id}, {"$set": {"status_ner": "Running"}})
    result = []
    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(text)
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

    jobCollection.update_one({"_id": ObjectId(job_id)},
                             {"$set": {"status_ner": "Completed", "end_time": datetime.now()}})

    logger.info(f"Analysis of text '{text}' has been completed")


@app.task
def analyze_sentiment_text(text, job_id):
    logger.info(f"job_id: {job_id}")
    jobCollection.update_one({"_id": job_id}, {"$set": {"status_sentiment": "Running"}})

    result = []
    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(text)
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

    jobCollection.update_one({"_id": ObjectId(job_id)},
                             {"$set": {"status_sentiment": "Completed", "end_time": datetime.now()}})

    logger.info(f"Sentiment analysis of text '{text}' has been completed")


@app.task
def analyze_text_using_custom_model(text, job_id):
    logger.info(f"job_id: {job_id}")
    jobCollection.update_one({"_id": job_id}, {"$set": {"status_ner": "Running"}})
    result = []
    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(text)
    for sentence in sentences:
        raw_result = analyze_custom_model(sentence.text)
        for word, label in raw_result:
            # Remove word starting with [ and ending with ] and remove punct
            if re.match(r"^\[.*\]$", word) or any(char in word for char in punctuation):
                logger.info(f"Skipping word: {word}")
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

    jobCollection.update_one({"_id": ObjectId(job_id)},
                             {"$set": {"status_ner": "Completed", "end_time": datetime.now()}})


if __name__ == "__main__":
    print(analyze_text_using_custom_model("Michael graduated from MIT in 2010. The MIT university is in Paris and USA",
                                          "1"))
