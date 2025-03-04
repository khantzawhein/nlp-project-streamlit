import streamlit as st
from bson import ObjectId

from db.mongo import get_db
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px


def render():
    st.set_page_config(
        page_title="Reports",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("Reports")

    param_id = st.query_params.get("id")

    db = get_db()
    collection = db["analysis_results"]

    reports = collection.find().sort("_id", -1).to_list()
    df = pd.DataFrame(reports)
    st.text("All reports:")
    if df.empty:
        st.warning("No reports found.")
        return
    st.dataframe(df[["job_id", "model", "text"]], hide_index=True)

    st.divider()

    st.write("Please select a report to open:")
    index = 0
    if param_id:
        try:
            index = [str(report["job_id"]) for report in reports].index(param_id)
        except ValueError:
            pass
    job_id = st.selectbox("Select a report", [str(report["job_id"]) for report in reports], index=index)
    st.write(f"Selected report: {job_id}")

    report = collection.find_one({"job_id": ObjectId(job_id)})
    if report:
        analysis = report["analysis"]

        with st.expander("Original Text"):
            st.write(report['text'])

        ner_analysis(analysis, report)
    else:
        st.warning("Analysis has not been completed yet.")

    st.markdown(f"<h4>Sentiment Analysis: </h4>", unsafe_allow_html=True)

    sentiment_collection = db["sentiment_results"]

    sentiment_report = sentiment_collection.find_one({"job_id": ObjectId(job_id)})

    if sentiment_report:
        sentiment_df = pd.DataFrame(sentiment_report['sentiment'])
        st.write(sentiment_df)
        col1, col2 = st.columns(2)
        with col1:
            plt.figure(figsize=(6, 6))
            plt.title("Sentiment Pie Chart")
            sentiment_df["sentiment"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=140,
                                                          colors=["#ff9999", "#66b3ff", "#99ff99"])
            st.pyplot(plt, clear_figure=True)
        with col2:
            # Bar Chart
            plt.figure(figsize=(6, 6))
            plt.title("Sentiment Distribution")
            sentiment_df["sentiment"].value_counts().plot(kind="bar")
            st.pyplot(plt, clear_figure=True)







    else:
        st.warning("Sentiment analysis has not been completed yet.")


def ner_analysis(analysis, report):
    st.markdown(f"<h4>Named-Entity Analysis: </h4>", unsafe_allow_html=True)
    analysis_df = pd.DataFrame(analysis)
    type_count = analysis_df["type"].value_counts()
    entity_count = analysis_df["text"].value_counts().sort_values(ascending=False)
    col1, col2 = st.columns(2)
    with col1:
        get_entity_type_distribution_chart(type_count)
    with col2:
        get_entity_bar_chart(entity_count)
    st.divider()
    get_word_cloud(analysis_df)
    st.divider()
    get_sunburst_chart(analysis_df, report, type_count)
    col3, col4 = st.columns([2, 3])
    with col3:
        st.write(type_count)
    with col4:
        st.write(analysis_df)
    with st.expander("Raw Analysis Result"):
        st.write(report)


def get_entity_bar_chart(entity_count):
    plt.figure(figsize=(6, 6))
    plt.title("Top Entities by Frequency")
    entity_count.head(10).plot(kind="bar")
    st.pyplot(plt, clear_figure=True)


def get_sunburst_chart(analysis_df, report, type_count):
    st.caption("Sunburst Chart")
    fig = px.sunburst(analysis_df, path=["type", "text"], width=800, height=800)
    st.plotly_chart(fig)


def get_word_cloud(analysis_df):
    st.caption("Word Cloud")
    text = " ".join(analysis_df["text"])
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt, clear_figure=True)


def get_entity_type_distribution_chart(type_count):
    plt.figure(figsize=(6, 6))
    plt.title("Entity Type Distribution")
    type_count.plot(kind="pie", autopct="%1.1f%%", startangle=140,
                    colors=["#ff9999", "#66b3ff", "#99ff99"])
    st.pyplot(plt, clear_figure=True)


render()
