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
            background: none !important;
            border: none !important;
            padding: 0 !important;
            cursor: pointer;
        }}

        /* Normal Text Style */
        div[role="radiogroup"] label div {{
            color: rgba(255, 255, 255, 0.7) !important; /* Fixed RGBA syntax */
            font-size: 18px !important;
            transition: 0.3s;
        }}

        /* Hover and Selected Style */
        div[role="radiogroup"] label:hover div,
        div[role="radiogroup"] label:has(input:checked) div {{
            color: white !important; 
            font-weight: bold !important;
            text-decoration: underline; 
            text-underline-offset: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    def set_home_background():
    st.markdown(
        f"""
        <style> 
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
            background: none !important;
            border: none !important;
            padding: 0 !important;
            cursor: pointer;
        }}

        /* Normal Text Style */
        div[role="radiogroup"] label div {{
            color: rgba(255, 255, 255, 0.7) !important; /* Fixed RGBA syntax */
            font-size: 18px !important;
            transition: 0.3s;
        }}

        /* Hover and Selected Style */
        div[role="radiogroup"] label:hover div,
        div[role="radiogroup"] label:has(input:checked) div {{
            color: white !important; 
            font-weight: bold !important;
            text-decoration: underline; 
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

    set_home_background()
    # =====================================
    # SECTION 1 : MAIN TITLE (NO IMAGE)
    # =====================================
    st.markdown("""
    <div style="
        width:100vw;
        margin-left:calc(-50vw + 50%);
        background-image: url('static/header2.png');
        background-size:cover;
        background-attachment: fixed;
        background-position:center;
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

    set_background("https://i.pinimg.com/1200x/78/1d/4c/781d4c6becbd05f20e26057f6cbaf9bc.jpg")

    st.title("Fraud Dashboard")

    embed_tableau(TABLEAU_PATHS["Fraud Overview"])



# =================================================
# TAB 3 EDA
# =================================================
elif nav == "Exploratory Data Analysis (EDA)":

    set_background("https://i.pinimg.com/1200x/78/1d/4c/781d4c6becbd05f20e26057f6cbaf9bc.jpg")

    st.title("Exploratory Data Analysis")
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

    col1, col2, col3, col4 = st.columns(4)

    with col1:
         with st.container(key="eda_distribution"):
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
    with col2:
        with st.container(key="eda_distribution_numerical"):
            # Chart 2
            st.subheader("Numerical Features Distribution")

            num_cols = ['transaction_amount', 'log_amount', 'hour', 'day', 'month', 'dayofweek', 'avg_monthly_spend']

            for col in num_cols:

                non_fraud = eda_df[eda_df['is_fraud'] == 0][col]
                fraud = eda_df[eda_df['is_fraud'] == 1][col]

                fig = make_subplots(
                    rows=1,
                    cols=2,
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
                    row=1,
                    col=1
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
                    row=1,
                    col=1
                )

                fig.add_trace(
                    go.Box(
                        y=non_fraud,
                        name="Non-fraud",
                        marker_color='steelblue',
                        boxmean=True
                    ),
                    row=1,
                    col=2
                )

                fig.add_trace(
                    go.Box(
                        y=fraud,
                        name="Fraud",
                        marker_color='crimson',
                        boxmean=True
                    ),
                    row=1,
                    col=2
                )

                fig.update_layout(
                    title_text=f"{col} Analysis by Fraud",
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
                **Key Findings:**

                **Transaction Amount vs Fraud**

                - Overlapping distributions between fraud and non-fraud
                - Similar medians and quartiles
                - Non-fraud has larger outliers

                **Log Amount vs Fraud**

                - Normalized distribution for both fraud and non-fraud centered around 7.5 to 8.0
                - No strong separation and confirming that transactiob amount alone is not a strong differentiator for detecting fraud.

                **Hour vs. Fraud**

                - Slight Evening Peak: There is a noticeable increase in fraud density during the evening hours (approx. 17:00 to 22:00).
                - Uniformity: Generally, both classes are distributed across all 24 hours, but fraud appears slightly more "concentrated" in certain blocks than non-fraud.

                **Day vs. Fraud**

                - Random Distribution: Fraud occurs fairly consistently throughout the month.
                - Minor Fluctuations: There are small spikes around day 5 and day 20, but the overall distributions (boxplots) for fraud and non-fraud are virtually identical.

                **Month vs. Fraud**

                - Late Year Surge: There is a distinct increase in the proportion of fraud during the later months, specifically months 7, 8, and 9.
                - Potential Seasonality: The boxplot for fraud shows a higher median month compared to non-fraud, suggesting fraud activity may increase as the year progresses.

                **Day of Week vs. Fraud**

                - Weekend Spike: Fraudulent transactions show a slight peak on Day 4 and Day 5 (Friday/Saturday).
                - Stability: Despite the minor weekend peak, the median and interquartile ranges are almost identical for both groups across the week.

                **Average Monthly Spend**

                - Consistent Behavior: The spending habits of users who were victims of fraud are nearly identical to those who were not.
                - No Financial Divergence: Both groups show a distribution peak around 5k, indicating that "high spenders" are not necessarily more or less targeted than "low spenders" in this dataset.
            """)
    with col3:
        with st.container(key="eda_distribution_cat"):
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
                st.plotly_chart(fig, use_container_width=True)   

            st.markdown("""

        **Key Findings:**

        **Fraud Rate by Payment Channel**

        - Highest risk: Card
        - Followed by: UPI
        - Lowest risk: Wallet

        **Fraud Rate by Device Type**
        - Highest Risk: Mobile and desktop devices show nearly identical, elevated fraud rates.
        - Lowest Risk: Tablets have a slightly lower fraud rate compared to the other two device types.
        - Consistency: The device type does not appear to be a drastic differentiator, as all rates remain near the 0.016 mark.

        **Fraud Rate by Is_Weekend

        - Weekend Spike: Fraud rates are notably higher on the weekend (is_weekend = 1) compared to weekdays.
        - Weekday Baseline: Transactions occurring during the week (is_weekend = 0) have a fraud rate of approximately 0.016, while weekends push toward 0.017.

        **Fraud Rate by Is_International**

        - Major Differentiator: This is the most significant factor shown; international transactions (is_international = 1) have a much higher fraud rate than domestic ones.
        - Risk Magnitude: The international fraud rate (approx. 0.034) is more than double the domestic fraud rate (approx. 0.015).
        """)
    with col4:
        with st.container(key="eda_time_analysis"):
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

            fig.update_layout(
                showlegend=False,
                title_text="Fraud Rate Analysis Over Time"
            )

            fig.update_yaxes(title_text="Fraud Rate", row=1, col=1)
            fig.update_yaxes(title_text="Fraud Rate", row=1, col=2)

            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""

        **Key Findings:**
        **Hourly Fraud Pattern**

        - Peak between 6pm-7pm
        - Early morning spike around 2am
        - Lowest around 9am

        **Transactional Time Analysis: By Date**

        - Upward Trend: The fraud rate shows a general increasing trend as the year progresses from January 2023 toward September 2023
        - High-Frequency Volatility: The "sawtooth" pattern indicates that fraud occurs in sharp, inconsistent bursts rather than a steady stream.
        - Significant Surge: A noticeable shift to a higher baseline fraud rate occurs around July 2023, with the highest peaks reaching nearly 0. 3 in late August/early September.
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

    set_background("https://i.pinimg.com/originals/7d/f1/43/7df143904208d2295c8428caaf86cb3c.gif")

    st.title("ML Fraud Detection")

    with st.container(key="main_container"):

        sidebar, main_page = st.columns([1,2])

        # =========================
        # SIDEBAR
        # =========================
        with sidebar:
            with st.container(key="sidebar"):
            
                st.image("https://i.pinimg.com/1200x/68/fd/7b/68fd7b646d8f0b18ab50204dd32c807f.jpg")
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

    set_background("https://i.pinimg.com/1200x/78/1d/4c/781d4c6becbd05f20e26057f6cbaf9bc.jpg")

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
