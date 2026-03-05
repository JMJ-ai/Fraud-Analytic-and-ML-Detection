import streamlit as st
import pandas as pd
import numpy as np
import yaml
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import zipfile
import urllib.request
from sqlalchemy import create_engine
import os

from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay

# -----------------------------
# Load config
# -----------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

USE_DATABASE = config["data"]["use_database"]

DATA_DIR = "data"
TRAIN_CSV = f"{DATA_DIR}/transactions_train.csv"
TEST_CSV = f"{DATA_DIR}/transactions_test.csv"

TRAIN_URL = config["files"]["train_url"]
TEST_URL = config["files"]["test_url"]


# -----------------------------
# Download dataset if missing
# -----------------------------
def download_and_extract(url, filename):

    os.makedirs(DATA_DIR, exist_ok=True)

    zip_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(zip_path):

        st.info(f"Downloading {filename}...")

        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

        st.success(f"{filename} downloaded and extracted.")


# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():

    if USE_DATABASE:

        try:

            db = config["database"]

            # CREATE ENGINE FIRST
            engine = create_engine(
                f"postgresql+psycopg2://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['dbname']}"
            )

            train_df = pd.read_sql(
                f"SELECT * FROM {config['tables']['train']}",
                engine
            )

            test_df = pd.read_sql(
                f"SELECT * FROM {config['tables']['test']}",
                engine
            )

            st.success("Connected to PostgreSQL")

        except Exception as e:

            st.warning("Database connection failed. Loading CSV instead.")

            download_and_extract(TRAIN_URL, "transaction_train.zip")
            download_and_extract(TEST_URL, "transaction_test.zip")

            train_df = pd.read_csv(TRAIN_CSV)
            test_df = pd.read_csv(TEST_CSV)

    else:

        download_and_extract(TRAIN_URL, "transaction_train.zip")
        download_and_extract(TEST_URL, "transaction_test.zip")

        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV)

    return train_df, test_df


train_df, test_df = load_data()
train_df["transaction_time"] = pd.to_datetime(train_df["transaction_time"])
test_df["transaction_time"] = pd.to_datetime(test_df["transaction_time"])

# -------------------------------
# Load models
# -------------------------------

model = joblib.load(config["model"]["path"])
preprocessor = joblib.load(config["model"]["preprocessor"])
selector = joblib.load(config["model"]["selector"])

# -------------------------------
# Streamlit page config
# -------------------------------

st.set_page_config(
    page_title="Fraud Analytics and ML Detection",
    layout="wide"
)

# -------------------------------
# Navbar
# -------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Fraud Dashboard",
        "EDA by Python",
        "ML Fraud Detection",
        "Model Evaluation",
        "Project Methodology"
    ]
)

# -------------------------------
# PAGE 1 DASHBOARD
# -------------------------------

if page == "Fraud Dashboard":

    st.title("Fraud Dashboard by Tableau")

    st.markdown(
        """
        <iframe
        src="YOUR_TABLEAU_EMBED_LINK"
        width="100%"
        height="700">
        </iframe>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# PAGE 2 EDA
# -------------------------------

elif page == "EDA by Python":

    st.title("Exploratory Data Analysis")

    # Chart 1 Target distribution
    st.subheader("Target Distribution")

    fraud_dist = train_df["is_fraud"].value_counts().reset_index()
    fraud_dist.columns = ["Class", "Count"]

    fig = px.bar(fraud_dist, x="Class", y="Count")

    st.plotly_chart(fig)

    st.markdown("""
**Key Findings**

• Extreme class imbalance  
• 98.4% non-fraud  
• 1.62% fraud transactions
""")

    # Chart 2 Transaction amount

    st.subheader("Transaction Amount Distribution")

    fig = px.box(
        train_df,
        x="is_fraud",
        y="transaction_amount"
    )

    st.plotly_chart(fig)

    # Chart 3 Payment channel fraud rate

    st.subheader("Fraud Rate by Payment Channel")

    channel = (
        train_df
        .groupby("payment_channel")["is_fraud"]
        .mean()
        .reset_index()
    )

    fig = px.bar(channel,
                 x="payment_channel",
                 y="is_fraud",
                 labels={
                     "payment_channel": "Payment Channel",
                     "is_fraud": "Fraud Rate"
                 })

    st.plotly_chart(fig)

# -------------------------------
# PAGE 3 ML DETECTION
# -------------------------------

elif page == "ML Fraud Detection":

    # Animated background
    st.markdown(
        """
        <style>
        .stApp {
        background: linear-gradient(-45deg,#1e3c72,#2a5298,#1e3c72);
        background-size: 400% 400%;
        animation: gradient 12s ease infinite;
        }

        @keyframes gradient {
        0% {background-position:0% 50%}
        50% {background-position:100% 50%}
        100% {background-position:0% 50%}
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("ML Fraud Detection")

    col1, col2 = st.columns(2)

    # ---------------------------
    # Model info
    # ---------------------------

    with col1:

       
        st.subheader("Prediction Model Info")

        st.write(f"Algorithm: {config['model']['algorithm']}")
        st.write(f"AUC Score: {config['model']['auc_score']}")

    # ---------------------------
    # Prediction tool
    # ---------------------------

    with col2:

        st.image("assets/predict_icon.png", width=120)

        st.subheader("Prediction Tool")

        amount = st.number_input("Transaction Amount")

        ip_risk = st.slider("IP Risk Score", 0.0, 1.0)

        merchant_risk = st.slider("Merchant Risk Score", 0.0, 1.0)

        account_age = st.number_input("Account Age Days")

        geo_dist = st.number_input("Geo Distance")

        txn_24h = st.number_input("Txn Count 24h")

        failed_txn = st.number_input("Failed Txn 24h")

        txn_1h = st.number_input("Txn Count 1h")

        avg_spend = st.number_input("Avg Monthly Spend")

        dev_amount = st.number_input("Amount Deviation")

        if st.button("Predict Fraud"):

            input_df = pd.DataFrame({
                "transaction_amount":[amount],
                "ip_risk_score":[ip_risk],
                "merchant_risk_score":[merchant_risk],
                "account_age_days":[account_age],
                "geo_distance_from_last_txn":[geo_dist],
                "txn_count_24h":[txn_24h],
                "failed_txn_count_24h":[failed_txn],
                "txn_count_1h":[txn_1h],
                "avg_monthly_spend":[avg_spend],
                "amount_deviation_from_user_mean":[dev_amount]
            })

            X = preprocessor.transform(input_df)
            X = selector.transform(X)

            prob = model.predict_proba(X)[0][1]

            st.metric("Fraud Probability", f"{prob*100:.2f}%")

            if prob > 0.7:
                st.error("⚠️ Oh no! It's fraud!")
            else:
                st.success("🟢 Phew! Not fraud")

# -------------------------------
# PAGE 4 MODEL EVALUATION
# -------------------------------

elif page == "Model Evaluation":

    st.title("Model Evaluation")

    X_test = test_df.drop("is_fraud", axis=1)
    y_test = test_df["is_fraud"]

    X_proc = preprocessor.transform(X_test)
    X_proc = selector.transform(X_proc)

    pred_prob = model.predict_proba(X_proc)[:,1]

    auc = roc_auc_score(y_test, pred_prob)

    st.metric("ROC AUC Score", round(auc,3))

    # Confusion Matrix

    pred = model.predict(X_proc)

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred,
        ax=ax
    )

    st.pyplot(fig)

# -------------------------------
# PAGE 5 METHODOLOGY
# -------------------------------

elif page == "Project Methodology":

    st.title("Project Methodology")

    st.image("assets/ml_pipeline.png")
# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown(
"""
---
<div style="text-align:center">

Jenifer M Jues • 2026

<br>

<a href="https://github.com/JMJ-ai/Fraud-Analytic-and-ML-Detection" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="35">
</a>

<a href="https://www.linkedin.com/in/jenifermayangjues/" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="35">
</a>

<a href="https://icons8.com/" target="_blank">
    <img src="https://img.icons8.com/?size=100&id=ayJDJ6xQKgM6&format=png&color=000000" width="35">
</a>

</div>
""",
unsafe_allow_html=True
)

st.caption("End-to-End Fraud Detection System using Machine Learning and Analytics")
