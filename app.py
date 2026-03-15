import streamlit as st
import pandas as pd
import numpy as np
import yaml
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import zipfile
import urllib.request
from sqlalchemy import create_engine
import os
import json
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
import sys
# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Fraud Analytics and ML Detection",
    layout="wide"
)
# -------------------------------------------------
# AUTO GENERATE MISSING FEATURES
# -------------------------------------------------
def build_full_feature_set(input_df):

    df = input_df.copy()

    # Derived features
    df["log_amount"] = np.log1p(df["transaction_amount"])

    # Time-based defaults
    now = pd.Timestamp.now()
    df["hour"] = now.hour
    df["day"] = now.day
    df["month"] = now.month
    df["dayofweek"] = now.dayofweek
    df["is_weekend"] = int(now.dayofweek >= 5)
    df["date"] = now.date()
    # Other defaults
    df["is_international"] = 0
    df["kyc_level"] = 1
    df["credit_score_band"] = 2

    return df
# -------------------------------------------------
# LOAD CONFIG
# -------------------------------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

USE_DATABASE = config["data"]["use_database"]

DATA_DIR = "data"
TRAIN_CSV = f"{DATA_DIR}/transactions_train.csv"

TRAIN_URL = config["files"]["train_url"]

def safe_to_float(value):
    try:
        return float(int(value))
    except:
        return 0.0
def safe_to_int(value):
    try:
        return int(value)
    except:
        return 0


import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
header_base64 = get_base64_image("assests/header2.png")
# -------------------------------------------------
# BACKGROUND IMAGE FUNCTION
# ------------------------------------------------    
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        /* 1. Background and Header Setup */
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-attachment: fixed;
        }}
        
        /* Hides header/footer and removes the top padding gap */
        header {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        .block-container {{
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
        }}

        /* 2. Remove radio bullets */
        div[role="radiogroup"] > label > div:first-child {{
            display: none !important;
        }}

        /* 3. Style text as simple titles */
        div[role="radiogroup"] {{
            gap: 30px; 
            justify-content: center;
        }}

        div[role="radiogroup"] label {{
            background: white !important;
            border: none !important;
            padding: 0 !important;
            cursor: pointer;
        }}

        /* Normal Text Style */
        div[role="radiogroup"] label div {{
            color: black !important; /* Fixed RGBA syntax */
            font-size: 18px !important;
            transition: 0.3s;
        }}

        /* Hover and Selected Style */
        div[role="radiogroup"] label:hover div,
        div[role="radiogroup"] label:has(input:checked) div {{
            color: #ff2b2b !important; 
            font-weight: bold !important;
            text-decoration: underline; 
            text-decoration-color:#ff2b2b;
            text-underline-offset: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# -------------------------------------------------
# DOWNLOAD DATA
# -------------------------------------------------
def download_and_extract(url, filename):

    os.makedirs(DATA_DIR, exist_ok=True)

    zip_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(zip_path):

        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():

    if USE_DATABASE:

        try:

            db = config["database"]

            engine = create_engine(
                f"postgresql+psycopg2://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['dbname']}"
            )

            train_df = pd.read_sql(
                f"SELECT * FROM {config['tables']['train']}",
                engine
            )

        except:

            download_and_extract(TRAIN_URL,"transaction_train.zip")

            train_df = pd.read_csv(TRAIN_CSV)

    else:

        download_and_extract(TRAIN_URL,"transaction_train.zip")
        
        train_df = pd.read_csv(TRAIN_CSV)
        
    return train_df

train_df = load_data()

# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------
train_df = train_df.sort_values("transaction_time")
train_df["transaction_time"] = pd.to_datetime(train_df["transaction_time"])

for df in [train_df]:

    df["hour"] = df["transaction_time"].dt.hour
    df["day"] = df["transaction_time"].dt.day
    df["month"] = df["transaction_time"].dt.month
    df["dayofweek"] = df["transaction_time"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >=5).astype(int)

train_df["log_amount"] = np.log1p(train_df["transaction_amount"])
bool_cols = ['is_fraud', 'is_international']
for col in bool_cols:
    train_df[col] = train_df[col].astype(int)
    
ordinal_cols = ['kyc_level', 'credit_score_band']

for col in ordinal_cols:
    train_df[col] = train_df[col].astype(int)
   
# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
pipeline = joblib.load(config["model"]["pipeline"])
training_columns = joblib.load(config["model"]["training_columns"])
# =====================================================
TABLEAU_PATHS = {
    "Fraud Overview": "views/FraudVisualizationAnalyticv1/FraudOverview"
}
# =====================================================
# SAFE TABLEAU EMBED FUNCTION
# =====================================================
def embed_tableau(path, height=650):
    html_code = f"""
    <script type='module' src='https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js'></script>
    <tableau-viz
        src="https://public.tableau.com/{path}"
        width="100%"
        height="{height}"
        toolbar="hidden"
        hide-tabs>
    </tableau-viz>
    """
    st.components.v1.html(html_code, height=height)
# -------------------------------------------------
# TABS NAVIGATION
# -------------------------------------------------
nav = st.radio(
    "",
    ["Home","Dashboard","Exploratory Data Analysis (EDA)","ML Detection", "Methodology"],
    horizontal=True,
    label_visibility="collapsed"
)
# =================================================
# HOME
# =================================================
if nav == "Home":
    set_background("https://i.pinimg.com/1200x/73/12/d4/7312d47a09137b32e094d33257096209.jpg")
    # =====================================
    # SECTION 1 : MAIN TITLE (NO IMAGE)
    # =====================================
    st.markdown("""
    <div style="
        flex:1;
        min-width:3009x
        padding:120px 20px;
        text-align:center;
    ">

    <h1 style="font-size:90px;color:red;">
    Fraud Analytics and ML Detection
    </h1>

    <h3 style="color:black;">
    Machine Learning Forecast & Prediction
    </h3>

    </div>

    <!-- RIGHT SIDE : IMAGE -->
    <div style="flex:1; min-width:300px; text-align:center;">

    <img src="https://i.pinimg.com/736x/b7/d2/c8/b7d2c894de9f7184f1b42c42d2cc9023.jpg"
    style="
        width:670px;
        border-radius:15px;
        box-shadow:0px 6px 25px rgba(0,0,0,0.6);
        ">

    </div>
    """, unsafe_allow_html=True)


    # =====================================
    # SECTION 2 : ABOUT PROJECT
    # IMAGE + 40% DARK OVERLAY
    # =====================================
    st.markdown("""
    <div style="
        width:100vw;
        margin-left:calc(-50vw + 50%);
        background-color:#062e0c;
        background-size:cover;
        background-position:center;    
    ">

    <div style="
        max-width:1000px;
        margin:auto;
        padding:80px 20px;
        text-align:center;
    ">

    <h2 style="text-align:center;color:white;">
    About This Project
    </h2>

    <br>

    <p style="color:white;font-size:18px;text-align:center;max-width:900px;margin:auto;">
    Financial fraud is a growing threat, costing banks and e-commerce platforms billions of dollars annually.
    Recent studies show ensemble machine learning models, such as CatBoost and XGBoost, can achieve a 
    ROC-AUC of ~0.84 on temporal transaction datasets, highlighting the importance of feature engineering 
    and robust model pipelines.
    </p>

    <br>

    <p style="color:white;font-size:18px;text-align:center;max-width:900px;margin:auto;">
    The goal is to explores feature selection to reduce redundancy, baseline modeling with LightGBM
    model exploration using Random Forest, AdaBoost, and XGBoost and feature importance analysis to interpret results
    </p>

    <br>

    <p style="text-align:center;color:white;">
    Data source:
    <a href="(https://www.kaggle.com/code/rohit8527kmr7518/fraud-detection-eda-modelling-0-84-auc/input" target="_blank" style="color:#FFD700;">Kaggle</a>

    </div>
    """, unsafe_allow_html=True)



    # =====================================
    # SECTION 3 : PROBLEM STATEMENT
    # DARK BACKGROUND
    # =====================================
    st.markdown("""
    <div style="
        width:100vw;
        margin-left:calc(-50vw + 50%);
        background-color:#6e460e;
    ">

    <div style="
        max-width:1000px;
        margin:auto;
        padding:80px 20px;
        text-align:center;
    ">

    <h2 style="text-align:center;color:white;">
    Problem Statement
    </h2>

    <br>

    <p style="color:white;font-size:18px;text-align:center;max-width:900px;margin:auto;">
    Detecting fraudulent transactions is challenging due to class imbalance, temporal drift, and noisy data. Traditional models
    often fail when minority fraud patterns are overshadowed by the majority of legitimate transactions
    </p>

    </div>
    """, unsafe_allow_html=True)
# =================================================
# TAB 2 DASHBOARD
# =================================================
elif nav == "Dashboard":

    set_background("https://i.pinimg.com/736x/d4/6f/a1/d46fa14d874ee0be170864d08227ccc8.jpg")

    st.title("Fraud Dashboard")

    embed_tableau(TABLEAU_PATHS["Fraud Overview"])



# =================================================
# TAB 3 EDA
# =================================================
elif nav == "Exploratory Data Analysis (EDA)":

    set_background("https://i.pinimg.com/736x/d4/6f/a1/d46fa14d874ee0be170864d08227ccc8.jpg")

    st.title("Exploratory Data Analysis")
    eda_df = train_df.copy()
    eda_df["log_amount"] = np.log1p(eda_df["transaction_amount"])
    st.markdown("""
    <style>

    /* White glass container */
    [class*="st-key-eda_"], 
    [class*="st-key-cluster_"] {

        background: rgba(255,255,255,0.40);
        backdrop-filter: blur(8px);

        padding: 25px;
        border-radius: 16px;

        border: 1px solid rgba(255,255,255,0.6);

        box-shadow: 
            0px 6px 25px rgba(0,0,0,0.35);

        transition: all 0.25s ease;
    }

    /* Hover animation */
    [class*="st-key-eda_"]:hover,
    [class*="st-key-cluster_"]:hover {

        transform: translateY(-4px);
        box-shadow: 
            0px 10px 30px rgba(0,0,0,0.45);

    }

    </style>
    """, unsafe_allow_html=True)

    # =========================
    # Container 1
    # Target Distribution
    # =========================

    with st.container():

        st.subheader("Target Data Distribution")

        counts = eda_df["is_fraud"].value_counts()

        labels = ["Non Fraud","Fraud"]
        values = [counts.get(0,0), counts.get(1,0)]

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type":"bar"},{"type":"pie"}]],
            subplot_titles=("Fraud Count","Fraud Proportion")
        )

        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color=["steelblue","crimson"])
            ),
            row=1,col=1
        )

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=["steelblue","crimson"]),
                textinfo="percent+label"
            ),
            row=1,col=2
        )

        st.plotly_chart(fig,use_container_width=True)

        st.markdown("""
**Key Findings**

- Extreme class imbalance  
- 98.4% transactions are non-fraud  
- Only 1.62% are fraud  
""")

    # =========================
    # Container 2
    # Numerical Distribution
    # =========================

    with st.container():

        st.subheader("Numerical Features Distribution")

        num_cols = [
            "transaction_amount",
            "log_amount",
            "hour",
            "day",
            "month",
            "dayofweek",
            "avg_monthly_spend"
        ]

        for col in num_cols:

            non_fraud = eda_df[eda_df["is_fraud"]==0][col]
            fraud = eda_df[eda_df["is_fraud"]==1][col]

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(f"{col} distribution",f"{col} boxplot")
            )

            fig.add_trace(
                go.Histogram(
                    x=non_fraud,
                    nbinsx=40,
                    opacity=0.6,
                    marker_color="steelblue",
                    name="Non Fraud"
                ),
                row=1,col=1
            )

            fig.add_trace(
                go.Histogram(
                    x=fraud,
                    nbinsx=40,
                    opacity=0.6,
                    marker_color="crimson",
                    name="Fraud"
                ),
                row=1,col=1
            )

            fig.add_trace(
                go.Box(
                    y=non_fraud,
                    name="Non Fraud",
                    marker_color="steelblue"
                ),
                row=1,col=2
            )

            fig.add_trace(
                go.Box(
                    y=fraud,
                    name="Fraud",
                    marker_color="crimson"
                ),
                row=1,col=2
            )

            st.plotly_chart(fig,use_container_width=True)

        st.markdown("""
**Key Findings**

- Transaction amount distributions overlap strongly.
- Fraud activity increases slightly in evening hours.
- Fraud shows a mild rise during later months of the year.
""")

    # =========================
    # Container 3
    # Categorical Distribution
    # =========================

    with st.container():

        st.subheader("Categorical Feature Distribution")

        cat_cols = [
            "payment_channel",
            "device_type",
            "is_weekend",
            "is_international"
        ]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=cat_cols
        )

        positions = [(1,1),(1,2),(2,1),(2,2)]

        for col,(r,c) in zip(cat_cols,positions):

            fraud_rate = (
                eda_df
                .groupby(col)["is_fraud"]
                .mean()
                .reset_index()
            )

            fig.add_trace(
                go.Bar(
                    x=fraud_rate[col],
                    y=fraud_rate["is_fraud"],
                    marker_color="indianred"
                ),
                row=r,
                col=c
            )

        fig.update_layout(
            height=600,
            showlegend=False
        )

        st.plotly_chart(fig,use_container_width=True)

        st.markdown("""
**Key Findings**

**Payment Channel**
- Highest risk: Card
- Lowest risk: Wallet

**Device Type**
- Mobile and desktop show similar fraud rates.

**Weekend**
- Slight increase in fraud during weekends.

**International Transactions**
- Fraud rate is more than double domestic transactions.
""")

    # =========================
    # Container 4
    # Time Based Analysis
    # =========================

    with st.container():

        st.subheader("Time Based Analysis")

        train_df["transaction_time"] = pd.to_datetime(train_df["transaction_time"])

        train_df["hour"] = train_df["transaction_time"].dt.hour
        train_df["date"] = train_df["transaction_time"].dt.date

        hourly = train_df.groupby("hour")["is_fraud"].mean().reset_index()
        daily = train_df.groupby("date")["is_fraud"].mean().reset_index()

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Fraud Rate by Hour","Fraud Rate by Date")
        )

        fig.add_trace(
            go.Scatter(
                x=hourly["hour"],
                y=hourly["is_fraud"],
                mode="lines",
                line=dict(color="royalblue")
            ),
            row=1,col=1
        )

        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["is_fraud"],
                mode="lines",
                line=dict(color="crimson")
            ),
            row=1,col=2
        )

        st.plotly_chart(fig,use_container_width=True)

        st.markdown("""
**Key Findings**

- Fraud peaks around evening hours (6-7 PM)
- Early morning spike around 2 AM
- Fraud rate increases slightly toward later months
""")
   
   
# =================================================
# TAB 4 ML DETECTION
# =================================================
elif nav == "ML Detection":

    st.markdown("""
    <style>

    /* MAIN DASHBOARD CONTAINER */
    .st-key-main_container{
        padding:30px;
        border-radius:15px;
        backdrop-filter: blur(6px);
    }

    /* SIDEBAR PANEL */
    .st-key-sidebar{
        background-color:#062906;
        padding:25px;
        border-radius:12px;
        min-height:450px;
        color:white;
    }

    /* MAIN CONTENT PANEL */
    .st-key-mainpanel{
        background-color:rgba(0,0,0,0);
        padding:25px;
        border-radius:12px;
        min-height:450px;
        color:white;
    }

    /* subtle card for results */
    .result-card{
        background:rgba(255,255,255,0.08);
        padding:20px;
        border-radius:12px;
        border:1px solid rgba(255,255,255,0.15);
        font-size:22px;
        text-align:center;
    }

    </style>
    """, unsafe_allow_html=True)

    set_background("https://i.pinimg.com/736x/d4/6f/a1/d46fa14d874ee0be170864d08227ccc8.jpg")

    st.title("ML Fraud Detection")

    with st.container(key="main_container"):

        sidebar, main_page = st.columns([1,2])

        # =========================
        # SIDEBAR
        # =========================
        with sidebar:
            with st.container(key="sidebar"):
            
                st.image("https://i.pinimg.com/1200x/fb/c2/24/fbc224806771400bb171344a8843036e.jpg")
                st.write("Fill details below to predict crop production")

                payment_channel=st.selectbox(
                "Payment Channel",
                ["card","upi","bank_transfer","wallet"]
                )

                device_type=st.selectbox(
                "Device Type",
                ["mobile","desktop","tablet"]
                )

                amount=st.text_input("Transaction Amount")
                ip_risk=st.text_input("IP Risk Score")
                merchant_risk=st.text_input("Merchant Risk Score")
                account_age=st.text_input("Account Age Days")
                geo_dist=st.text_input("Geo Distance")
                txn_24h=st.text_input("Txn Count 24h")
                failed_txn=st.text_input("Failed Txn 24h")
                txn_1h=st.text_input("Txn Count 1h")
                avg_spend=st.text_input("Avg Monthly Spend")
                dev_amount=st.text_input("Amount Deviation")

                predict_button = st.button("Predict Production")

        # =========================
        # MAIN PAGE
        # =========================
        with main_page:
            with st.container(key="mainpanel"):

                st.subheader("Prediction Model Info")        

                st.write("Algorithm:",config["model"]["algorithm"])
                st.write("AUC Score:",config["model"]["auc_score"])

                result_placeholder = st.empty()
                if predict_button:

                    amount = safe_to_float(amount)
                    ip_risk = safe_to_float(ip_risk)
                    merchant_risk = safe_to_float(merchant_risk)
                    geo_dist = safe_to_float(geo_dist)
                    avg_spend = safe_to_float(avg_spend)
                    dev_amount = safe_to_float(dev_amount)

                    account_age = safe_to_int(account_age)
                    txn_24h = safe_to_int(txn_24h)
                    failed_txn = safe_to_int(failed_txn)
                    txn_1h = safe_to_int(txn_1h)
    
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
            
                    input_df = build_full_feature_set(input_df)
                    input_df = input_df.reindex(columns=training_columns, fill_value=0)
            
                    pred = pipeline.predict(input_df)
                    prob = pipeline.predict_proba(input_df)[0][1]

                    result_placeholder.markdown(
                            f"""
                            <div class="result-card">
                            🌾 Fraud Probability<br><br>
                            <b>{prob*100:.2f}</b>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            
                    if prob>0.7:
                        st.error("⚠️ Oh no! It's fraud!")
                    else:
                        st.success("🟢 Phew! Not fraud")

# =================================================
# TAB 5 METHODOLOGY
# =================================================
elif nav == "Methodology":

    set_background("https://i.pinimg.com/736x/d4/6f/a1/d46fa14d874ee0be170864d08227ccc8.jpg")

    st.title("Project Methodology")

    with st.container(horizontal_alignment="center"):
        st.image("assests/flowchart.png", width=800)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
footer_html = """
<style>

.footer {
position: relative;
left: 0;
bottom: 0;
width: 100%;
background-color: transparent;
color: white;
text-align: center;
padding: 20px 0;
font-size: 14px;
z-index: 9999;
}

.footer a {
margin: 0 10px;
text-decoration: none;
}

.footer img {
width: 28px;
margin-left: 8px;
margin-right: 8px;
vertical-align: middle;
transition: transform 0.2s;
}

.footer img:hover {
transform: scale(1.2);
}

</style>

<div class="footer">

<p>
Built with Data & Passion | © 2026 Jenifer M Jues
</p>

<a href="https://github.com/JMJ-ai/Fraud-Analytic-and-ML-Detection" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/25/25231.png">
</a>

<a href="https://www.linkedin.com/in/jenifermayangjues/" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png">
</a>

<a href="https://icons8.com/" target="_blank">
<img src="https://img.icons8.com/?size=100&id=ayJDJ6xQKgM6&format=png&color=000000">
</a>

<a href="mailto:jeniferjues@gmail.com">
<img src="https://cdn-icons-png.flaticon.com/512/732/732200.png">
</a>

</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
