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

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Fraud Analytics and ML Detection",
    layout="wide"
)

# -------------------------------------------------
# LOAD CONFIG
# -------------------------------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

USE_DATABASE = config["data"]["use_database"]

DATA_DIR = "data"
TRAIN_CSV = f"{DATA_DIR}/transactions_train.csv"
TEST_CSV = f"{DATA_DIR}/transactions_test.csv"

TRAIN_URL = config["files"]["train_url"]
TEST_URL = config["files"]["test_url"]

# -------------------------------------------------
# BACKGROUND IMAGE FUNCTION
# -------------------------------------------------
def set_background(image_url):

    st.markdown(
        f"""
        <style>

        .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
        background-attachment: fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# PARTICLE JS BACKGROUND
# -------------------------------------------------
def particle_background():

    with open("assests/particles.json") as f:
        particles_config = json.load(f)

    st.markdown(
        """
        <div id="particles-js"></div>

        <style>
        #particles-js {
        position: fixed;
        width: 100%;
        height: 100%;
        z-index: -1;
        top: 0;
        left: 0;
        }
        </style>

        <script src="https://cdn.jsdelivr.net/npm/particles.js"></script>

        <script>
        particlesJS.load('particles-js', 'assests/particles.json');
        </script>
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

            test_df = pd.read_sql(
                f"SELECT * FROM {config['tables']['test']}",
                engine
            )

        except:

            download_and_extract(TRAIN_URL,"transaction_train.zip")
            download_and_extract(TEST_URL,"transaction_test.zip")

            train_df = pd.read_csv(TRAIN_CSV)
            test_df = pd.read_csv(TEST_CSV)

    else:

        download_and_extract(TRAIN_URL,"transaction_train.zip")
        download_and_extract(TEST_URL,"transaction_test.zip")

        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV)

    return train_df, test_df

train_df, test_df = load_data()

# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------
train_df["transaction_time"] = pd.to_datetime(train_df["transaction_time"])
test_df["transaction_time"] = pd.to_datetime(test_df["transaction_time"])

for df in [train_df,test_df]:

    df["hour"] = df["transaction_time"].dt.hour
    df["day"] = df["transaction_time"].dt.day
    df["month"] = df["transaction_time"].dt.month
    df["dayofweek"] = df["transaction_time"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >=5).astype(int)

train_df["log_amount"] = np.log1p(train_df["transaction_amount"])
test_df["log_amount"] = np.log1p(test_df["transaction_amount"])

bool_cols = ['is_fraud', 'is_international']
for col in bool_cols:
    train_df[col] = train_df[col].astype(int)
    test_df[col] = test_df[col].astype(int)

ordinal_cols = ['kyc_level', 'credit_score_band']

for col in ordinal_cols:
    train_df[col] = train_df[col].astype(int)
    test_df[col] = test_df[col].astype(int)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = joblib.load(config["model"]["path"])
preprocessor = joblib.load(config["model"]["preprocessor"])
selector = joblib.load(config["model"]["selector"])

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

# -------------------------------

# -------------------------------------------------
# TABS NAVIGATION
# -------------------------------------------------
tabs = st.tabs([
"Home",
"Fraud Overview",
"Exploratory Data Analysis (EDA)",
"ML Detection",
"Model Evaluation",
"Methodology"
])

# =================================================
# TAB 1 HOME
# =================================================
with tabs[0]:

    set_background("https://i.pinimg.com/736x/9d/be/f5/9dbef56b9bcec4174ace6442499b7bb0.jpg")

    col1,col2 = st.columns([1,2])

    with col1:
        st.image("assests/home_image.jpg", use_container_width=True)

    with col2:

        st.markdown(
        """
        <h1 style='font-size:60px;color:white'>
        Fraud Analytics & Machine Learning Detection
        </h1>

        <h3 style='color:white'>
        End-to-End Fraud Detection System
        </h3>
        """,
        unsafe_allow_html=True
        )

# =================================================
# TAB 2 DASHBOARD
# =================================================
with tabs[1]:

    set_background("https://i.pinimg.com/1200x/78/1d/4c/781d4c6becbd05f20e26057f6cbaf9bc.jpg")

    st.title("Fraud Dashboard")

    embed_tableau(TABLEAU_PATHS["Fraud Overview"])



# =================================================
# TAB 3 EDA
# =================================================
with tabs[2]:

    set_background("https://i.pinimg.com/1200x/78/1d/4c/781d4c6becbd05f20e26057f6cbaf9bc.jpg")

    st.title("Exploratory Data Analysis")

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
        st.plotly_chart(fig, use_container_width=True)   

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

    fig.update_layout(
        showlegend=False,
        title_text="Fraud Rate Analysis Over Time"
    )

    fig.update_yaxes(title_text="Fraud Rate", row=1, col=1)
    fig.update_yaxes(title_text="Fraud Rate", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
**Hourly Fraud Pattern**

- Peak between 6pm-7pm
- Early morning spike around 2am
- Lowest around 9am
""")

    
# =================================================
# TAB 4 ML DETECTION
# =================================================
with tabs[3]:

    particle_background()

    st.title("ML Fraud Detection")

    col1,col2=st.columns(2)

    with col1:

        st.image("assests/model_icon.png",width=120)

        st.subheader("Prediction Model Info")

        st.write("Algorithm:",config["model"]["algorithm"])
        st.write("AUC Score:",config["model"]["auc_score"])

    with col2:

        st.image("assests/predict_icon.png",width=120)

        st.subheader("Prediction Tool")

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

        if st.button("Predict Fraud"):

            input_df=pd.DataFrame({

            "transaction_amount":[float(amount)],
            "ip_risk_score":[float(ip_risk)],
            "merchant_risk_score":[float(merchant_risk)],
            "account_age_days":[int(account_age)],
            "geo_distance_from_last_txn":[float(geo_dist)],
            "txn_count_24h":[int(txn_24h)],
            "failed_txn_count_24h":[int(failed_txn)],
            "txn_count_1h":[int(txn_1h)],
            "avg_monthly_spend":[float(avg_spend)],
            "payment_channel":[payment_channel],
            "device_type":[device_type],
            "amount_deviation_from_user_mean":[float(dev_amount)]

            })

            input_df["log_amount"]=np.log1p(input_df["transaction_amount"])

            X=preprocessor.transform(input_df)
            X=selector.transform(X)

            prob=model.predict_proba(X)[0][1]

            st.metric("Fraud Probability",f"{prob*100:.2f}%")

            if prob>0.7:
                st.error("⚠️ Oh no! It's fraud!")
            else:
                st.success("🟢 Phew! Not fraud")

# =================================================
# TAB 5 MODEL EVALUATION
# =================================================
with tabs[4]:

    set_background("https://i.pinimg.com/1200x/78/1d/4c/781d4c6becbd05f20e26057f6cbaf9bc.jpg")

    st.title("Model Evaluation")

    X_test=test_df.drop("is_fraud",axis=1)
    y_test=test_df["is_fraud"]

    X_proc=preprocessor.transform(X_test)
    X_proc=selector.transform(X_proc)

    prob=model.predict_proba(X_proc)[:,1]

    auc=roc_auc_score(y_test,prob)

    st.metric("ROC AUC Score",round(auc,3))

    st.write("Test shape:", X_test.shape)
    st.write("Processed shape:", X_proc.shape)

    st.write("Sample probabilities:")
    st.write(prob[:10])


    pred=model.predict(X_proc)

    fig,ax=plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred,
        ax=ax
    )

    st.pyplot(fig)

# =================================================
# TAB 6 METHODOLOGY
# =================================================
with tabs[5]:

    set_background("https://i.pinimg.com/1200x/78/1d/4c/781d4c6becbd05f20e26057f6cbaf9bc.jpg")

    st.title("Project Methodology")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
footer_html = """
<style>

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: rgba(0,0,0,0.85);
color: white;
text-align: center;
padding: 10px;
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
