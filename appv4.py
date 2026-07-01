import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
from bertopic import BERTopic

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_artifacts():

    ticket_model = joblib.load(
        "ticket_classifier_model.pkl"
    )

    ticket_vectorizer = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    business_mapping = joblib.load(
        "business_mapping.pkl"
    )

    topic_model = joblib.load(
        "bertopic_model"
    )

    business_topic_labels = joblib.load(
        "business_topic_labels.pkl"
    )

    churn_model = joblib.load(
        "churn_model.pkl"
    )

    scaler = joblib.load(
        "scaler.pkl"
    )

    feature_columns = joblib.load(
        "feature_columns.pkl"
    )

    return (
        ticket_model,
        ticket_vectorizer,
        business_mapping,
        topic_model,
        business_topic_labels,
        churn_model,
        scaler,
        feature_columns
    )

(
    ticket_model,
    ticket_vectorizer,
    business_mapping,
    topic_model,
    business_topic_labels,
    churn_model,
    scaler,
    feature_columns
) = load_artifacts()

# ============================================================
# CANDIDATE 1
# ============================================================

def predict_ticket_category(text):

    transformed = ticket_vectorizer.transform(
        [text]
    )

    prediction = ticket_model.predict(
        transformed
    )[0]

    category = business_mapping.get(
        prediction,
        str(prediction)
    )

    high_priority_keywords = [

        "urgent",
        "critical",
        "refund",
        "cancel",
        "failed",
        "error",
        "unable"
    ]

    priority = "Low"

    if any(
        word in text.lower()
        for word in high_priority_keywords
    ):
        priority = "High"

    return category, priority

# ============================================================
# CANDIDATE 2
# ============================================================

def predict_sentiment_emotion(text):

    polarity = TextBlob(
        text
    ).sentiment.polarity

    if polarity > 0.1:

        sentiment = "Positive"

    elif polarity < -0.1:

        sentiment = "Negative"

    else:

        sentiment = "Neutral"

    text_lower = text.lower()

    if any(
        x in text_lower
        for x in [
            "angry",
            "terrible",
            "frustrated",
            "refund",
            "cancel"
        ]
    ):
        emotion = "Angry"

    elif any(
        x in text_lower
        for x in [
            "happy",
            "great",
            "excellent",
            "amazing",
            "love"
        ]
    ):
        emotion = "Happy"

    elif any(
        x in text_lower
        for x in [
            "sad",
            "disappointed"
        ]
    ):
        emotion = "Sad"

    else:
        emotion = "Neutral"

    return sentiment, emotion, polarity

# ============================================================
# CANDIDATE 3
# ============================================================

def predict_topic(text):

    topic, prob = topic_model.transform(
        [text]
    )

    topic_id = int(topic[0])

    topic_label = business_topic_labels.get(
        topic_id,
        "Unknown Topic"
    )

    confidence = 0

    if prob is not None:

        confidence = round(
            float(max(prob[0])),
            4
        )

    return topic_id, topic_label, confidence

# ============================================================
# CANDIDATE 4
# ============================================================

def build_churn_features(
    sentiment,
    sentiment_score,
    priority
):

    return pd.DataFrame([{

        "ticket_count": 1,

        "avg_sentiment":
            sentiment_score,

        "usage": 50,

        "Ticket Frequency":
            5 if priority == "High"
            else 2,

        "sentiment_category_Neutral":
            1 if sentiment == "Neutral"
            else 0,

        "sentiment_category_Positive":
            1 if sentiment == "Positive"
            else 0
    }])

def predict_churn(
    sentiment,
    sentiment_score,
    priority
):

    features = build_churn_features(
        sentiment,
        sentiment_score,
        priority
    )

    scaled_part = scaler.transform(

        features[
            [
                "ticket_count",
                "avg_sentiment",
                "usage"
            ]
        ]
    )

    scaled_df = pd.DataFrame(

        scaled_part,

        columns=[
            "ticket_count",
            "avg_sentiment",
            "usage"
        ]
    )

    scaled_df["Ticket Frequency"] = (
        features["Ticket Frequency"].values
    )

    scaled_df["sentiment_category_Neutral"] = (
        features[
            "sentiment_category_Neutral"
        ].values
    )

    scaled_df["sentiment_category_Positive"] = (
        features[
            "sentiment_category_Positive"
        ].values
    )

    scaled_df = scaled_df[
        churn_model.feature_names_in_
    ]

    churn_prob = (
        churn_model.predict_proba(
            scaled_df
        )[0][1]
    )

    if churn_prob >= 0.70:

        churn_risk = "High"

    elif churn_prob >= 0.40:

        churn_risk = "Medium"

    else:

        churn_risk = "Low"

    return churn_prob, churn_risk

# ============================================================
# RISK ENGINE
# ============================================================

def business_risk(
    priority,
    sentiment,
    churn_prob
):

    score = 0

    score += (
        40 if priority == "High"
        else 10
    )

    score += (
        35 if sentiment == "Negative"
        else 15
    )

    score += churn_prob * 25

    if score >= 75:

        level = "Critical"

    elif score >= 50:

        level = "High"

    else:

        level = "Medium"

    return round(score, 2), level

# ============================================================
# UI
# ============================================================

st.title(
    "📊 Customer Intelligence Platform"
)

user_text = st.text_area(
    "Enter Customer Ticket"
)

if st.button("Analyze Ticket"):

    if len(user_text.strip()) == 0:

        st.warning(
            "Please enter a ticket."
        )

    else:

        category, priority = (
            predict_ticket_category(
                user_text
            )
        )

        sentiment, emotion, sentiment_score = (
            predict_sentiment_emotion(
                user_text
            )
        )

        topic_id, topic_label, topic_confidence = (
            predict_topic(
                user_text
            )
        )

        churn_prob, churn_risk = (
            predict_churn(
                sentiment,
                sentiment_score,
                priority
            )
        )

        risk_score, risk_level = (
            business_risk(
                priority,
                sentiment,
                churn_prob
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Classification"
            )

            st.write(
                "Category:",
                category
            )

            st.write(
                "Priority:",
                priority
            )

            st.subheader(
                "Sentiment"
            )

            st.write(
                "Sentiment:",
                sentiment
            )

            st.write(
                "Emotion:",
                emotion
            )

            st.write(
                "Score:",
                round(
                    sentiment_score,
                    4
                )
            )

        with col2:

            st.subheader(
                "Topic Analysis"
            )

            st.write(
                "Topic ID:",
                topic_id
            )

            st.write(
                "Topic:",
                topic_label
            )

            st.write(
                "Confidence:",
                topic_confidence
            )

            st.subheader(
                "Churn Analysis"
            )

            st.write(
                "Probability:",
                round(
                    churn_prob,
                    4
                )
            )

            st.write(
                "Risk:",
                churn_risk
            )

        st.subheader(
            "Business Impact"
        )

        st.metric(
            "Business Risk Score",
            risk_score
        )

        st.metric(
            "Risk Level",
            risk_level
        )

        st.subheader(
            "Recommendations"
        )

        if risk_level == "Critical":

            st.error(
                "Immediate escalation required."
            )

        elif risk_level == "High":

            st.warning(
                "Prioritize customer retention."
            )

        else:

            st.success(
                "Normal monitoring recommended."
            )
