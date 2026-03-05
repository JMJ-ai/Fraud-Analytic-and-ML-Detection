![header](header.png)
[## Project Overview 

**Project Background**

Financial fraud is a growing threat, costing banks and e-commerce platforms billions of dollars annually. Recent studies show ensemble machine learning models, such as CatBoost and XGBoost, can achieve a ROC-AUC of ~0.84 on temporal transaction datasets, highlighting the importance of feature engineering and robust model pipelines.

This project explores:

- Feature selection to reduce redundancy

- Baseline modeling with LightGBM

- Model exploration using Random Forest, AdaBoost, and XGBoost

- Feature importance analysis to interpret results

**Tools & Tech**
- PostgreSQL (Data Store)

- Python (Pandas, NumPy, Scikit-learn, Plotly)

- Tableau (Dashboard storytelling)

- Streamlit (Interactive deployment)

**Methodology**
- EDA
- Feature Importance Analysis
- Model Development for Fraud Prediction


**Metadata**

- Data source: Transactions_train.csv & Transaction_test.csv at [Kaggle](https://www.kaggle.com/code/rohit8527kmr7518/fraud-detection-eda-modelling-0-84-auc/input)
- Size:
    - Train shape: (300113 rows, 21 columns)
    - Test shape: (99887 rows, 21 columns)

**Problem Statement**

Detecting fraudulent transactions is challenging due to class imbalance, temporal drift, and noisy data. Traditional models often fail when minority fraud patterns are overshadowed by the majority of legitimate transactions

**Real World Impact**

- Reduce financial losses due to fraud

- Improve customer trust and retention

- Enable banks and e-commerce platforms to monitor real-time transactions

- Serve as a deployable ML pipeline integrated with dashboards for operational monitoring
