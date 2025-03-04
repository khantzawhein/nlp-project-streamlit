import streamlit as st
import pandas as pd
from db.mongo import get_db

st.set_page_config(
    page_title="Jobs",
    layout="wide",
    initial_sidebar_state="expanded"
)

db = get_db()
jobCollection = db["jobs"]
jobs = jobCollection.find().sort({"_id": -1}).to_list()

# Print background running jobs from Mongo DB as table

st.title('Background Jobs')

# Get all jobs from MongoDB

# Print table with job id, status, and start time

st.dataframe(pd.DataFrame({
    "Text": [job["text"] for job in jobs],
    "Model": [job["model"] for job in jobs],
    "Status (NER)": [job["status_ner"] if "status_ner" in job else "Not Running" for job in jobs],
    "Status (Sentiment)": [job["status_ner"] for job in jobs],
    "Action": [f"/reports?id={str(job["_id"])}" for job in jobs],
}), column_config={
    "Action": st.column_config.LinkColumn(
        display_text="Open Report",
    )
})


