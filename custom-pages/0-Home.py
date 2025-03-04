from datetime import datetime

import streamlit as st
from tasks.analyze_text import analyze_text, analyze_sentiment_text, analyze_text_using_custom_model
from db.mongo import get_db

db = get_db()

collection = db["jobs"]
st.set_page_config(
    page_title="Home",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.title('Named Entity Recognition + Sentiment Analysis')
st.caption("This is a project of NLP Course at CAMT, Chiang Mai University.")

with st.form(clear_on_submit=True, key="1"):
    noti_box = st.empty()
    text = st.text_area("Enter text to analyze:", placeholder="Enter text here")
    selected_model = st.radio("Select Model", ("Flair", "Custom Trained"))
    if st.form_submit_button(':rocket: **Analyze**'):
        result = collection.insert_one({
            "text": text,
            "model": selected_model,
            "status_ner": "Pending",
            "status_sentiment": "Pending",
            "start_time": datetime.now()
        })
        noti_box.success("Analysis has started.", icon="🚀")
        if selected_model == "Flair":
            analyze_text.delay(text, str(result.inserted_id))
        else:
            analyze_text_using_custom_model.delay(text, str(result.inserted_id))

        analyze_sentiment_text.delay(text, str(result.inserted_id))
