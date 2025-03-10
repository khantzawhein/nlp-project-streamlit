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

st.title('Background Jobs')

if len(jobs) == 0:
    st.warning("No Jobs Found")
    st.stop()


st.dataframe(pd.DataFrame({
    "Text": [job["text"] for job in jobs],
    "Model": [job["model"] for job in jobs],
    "Status (NER)": [job["status_ner"] for job in jobs],
    "Status (Sentiment)": [job["status_sentiment"] for job in jobs],
    "Action": [f"/reports?id={str(job['_id'])}" for job in jobs],
}), column_config={
    "Action": st.column_config.LinkColumn(
        display_text="Open Report",
    )
})


