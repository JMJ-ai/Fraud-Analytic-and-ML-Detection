import streamlit as st
import pandas as pd
import numpy as np
import yaml
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        src="https://public.tableau.com/views/FraudVisualizationAnalyticv1/FraudOverview?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link"
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

    st.header("Exploratory Data Analysis by Python")

    eda_df = pd.DataFrame(train_df.copy())
    eda_df['log_amount'] = np.log1p(eda_df['transaction_amount'])

    # Chart 1
    st.subheader("Target Data Distribution")

    counts = eda_df['is_fraud'].value_counts()

    labels = ['Non Fraud', 'Fraud']
    values = [counts.get(0,0), counts.get(1,0)]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type":"bar"},{"type":"pie"}]],
        subplot_titles=("Fraud Count","Fraud Proportion")
    )

    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color=['steelblue', 'crimson'])
        ),           
        row=1,col=1
    )

    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=['steelblue', 'crimson']),
            textinfo='percent+label',
            rotation=90
        ),
        row=1,col=2
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Key Findings**

- Extreme Class Imbalance
- 98.4% transactions are non-fraud
- Only 1.62% are fraud
""")

    # Chart 2
    st.subheader("Numerical Features Distribution")

    num_cols = ['transaction_amount', 'log_amount', 'hour', 'day', 'month', 'dayofweek', 'avg_monthly_spend']


    for col in num_cols:
        non_fraud = eda_df[eda_df['is_fraud'] == 0][col]
        fraud = eda_df[eda_df['is_fraud'] == 1][col]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"{col} distribution by fraud",
            f"{col} boxplot by fraud"
        )
     )

    fig.add_trace(
        go.Histogram(
            x=non_fraud,
            nbinsx=50,
            name="Non-fraud",
            histnorm='probability density',
            opacity=0.6,
            marker_color='steelblue'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=fraud,
            nbinsx=50,
            name="Fraud",
            histnorm='probability density',
            opacity=0.6,
            marker_color='crimson'
        ),
        row=1, col=1
    )
    
    # Overlay histograms
    fig.update_layout(barmode='overlay')
    
    # Boxplots
    fig.add_trace(
        go.Box(
            y=non_fraud,
            name="Non-fraud",
            marker_color='steelblue',
            boxmean=True
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Box(
            y=fraud,
            name="Fraud",
            marker_color='crimson',
            boxmean=True
        ),
        row=1, col=2
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Transaction Amount vs Fraud**

- Overlapping distributions between fraud and non-fraud
- Similar medians and quartiles
- Non-fraud has larger outliers

**Log Amount vs Fraud**

- Normalized distribution
- No strong separation
""")

    # Chart 3
    st.subheader("Categorical Feature Distribution")

    cat_cols_eda = [
        'payment_channel', 
        'device_type', 
        'is_weekend',  
        'is_international'
    ]

    for col in cat_cols_eda:
    
# Calculate fraud rate
        fraud_rate = (
            eda_df
            .groupby(col)['is_fraud']
            .mean()
            .reset_index()
            .sort_values(by='is_fraud')
        )
    
        # Plot horizontal bar chart
        fig = px.bar(
            fraud_rate,
            x='is_fraud',
            y=col,
            orientation='h',
            color=col,                 # <-- color by column
            title=f'Fraud rate by {col}',
            labels={'is_fraud': 'Fraud rate'}
        )

    st.markdown("""
**Fraud Rate by Payment Channel**

- Highest risk: Card
- Followed by: UPI
- Lowest risk: Wallet
""")

    # Chart 4
    st.subheader("Time Based Analysis")
    train_df['transaction_time'] = pd.to_datetime(train_df['transaction_time'])

    # Create hour and date columns
    train_df['hour'] = train_df['transaction_time'].dt.hour
    train_df['date'] = train_df['transaction_time'].dt.date

    # Compute fraud rates
    hourly_fraud = train_df.groupby('hour')['is_fraud'].mean().reset_index()
    daily_fraud = train_df.groupby('date')['is_fraud'].mean().reset_index()

    # Create subplots (1 row, 2 columns)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Fraud Rate by Hour of Day", "Fraud Rate by Date")
    )

    # --- Hourly Fraud ---
    fig.add_trace(
        go.Scatter(
            x=hourly_fraud['hour'],
            y=hourly_fraud['is_fraud'],
            mode='lines',
            line=dict(color='royalblue', width=2),
            name='Hourly Fraud Rate'
        ),
        row=1,
        col=1
    )

    # --- Daily Fraud ---
    fig.add_trace(
        go.Scatter(
            x=daily_fraud['date'],
            y=daily_fraud['is_fraud'],
            mode='lines',
            line=dict(color='crimson', width=2),
            name='Daily Fraud Rate'
        ),
        row=1,
        col=2
    )
    st.markdown("""
**Hourly Fraud Pattern**

- Peak between 6pm-7pm
- Early morning spike around 2am
- Lowest around 9am
""")

# -------------------------------
# PAGE 3 ML DETECTION
# -------------------------------

elif page == "ML Fraud Detection":

    st.title("ML Fraud Detection")

    col1, col2 = st.columns(2)

    # ---------------------------
    # Model info
    # ---------------------------

    with col1:

        st.image("assests/model_icon.png", width=120)
        st.subheader("Prediction Model Info")

        st.write(f"Algorithm: {config['model']['algorithm']}")
        st.write(f"AUC Score: {config['model']['auc_score']}")

    # ---------------------------
    # Prediction tool
    # ---------------------------

    with col2:

        st.image("assests/predict_icon.png", width=120)

        st.subheader("Prediction Tool")

        payment_channel = st.selectbox(
            "Payment Channel",
            ["card","upi","bank_transfer","wallet"]
        )

        device_type = st.selectbox(
            "Device Type",
            ["mobile","desktop","tablet"]
        )

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
                "payment_channel":[payment_channel],
                "device_type":[device_type],
                "amount_deviation_from_user_mean":[dev_amount]

            })

            input_df["log_amount"] = np.log1p(input_df["transaction_amount"])

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
