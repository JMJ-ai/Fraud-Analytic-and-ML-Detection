![header](header.png)

## Project Overview

Financial fraud continues to pose a significant challenge to banks and e-commerce platforms, resulting in billions of dollars in losses annually. Recent studies have demonstrated that ensemble machine learning models, such as XGBoost and CatBoost, can achieve ROC-AUC scores of approximately 0.84 on temporal transaction datasets, emphasizing the importance of effective feature engineering and robust predictive modeling.

This project aimed to develop an efficient fraud detection framework by identifying the most relevant predictive features, evaluating the performance of multiple ensemble machine learning models, and determining the most effective model for fraud prediction.

To reduce feature redundancy and improve model efficiency, feature selection techniques were applied, resulting in the identification of 10 key features that significantly contribute to fraud detection: IP risk score, average monthly spend, account age (days), geographical distance from the previous transaction, transaction amount, transaction amount deviation from user mean, transaction count within 24 hours, failed transaction count within 24 hours, transaction count within 1 hour, and transaction velocity indicators.

The predictive performance of several ensemble learning algorithms was evaluated, including Random Forest, AdaBoost, and XGBoost, with LightGBM serving as the baseline model for comparison. Model performance was assessed using the Area Under the Receiver Operating Characteristic Curve (AUC-ROC), a widely used metric for fraud detection tasks.

The results indicate that AdaBoost achieved the highest predictive performance with an AUC score of 0.8322, making it the best-performing model among those evaluated. Although the achieved AUC score was slightly lower than that reported in a previous study using the same dataset (0.8322 vs. 0.837), this project successfully reduced the number of input features from 20 to 10 while maintaining comparable predictive accuracy.

Compared with previous research, this study demonstrates that effective feature selection can substantially reduce model complexity without significantly compromising performance. Unlike the previous study, which utilized XGBoost and CatBoost without feature selection, this project employed a feature selection approach and evaluated AdaBoost and XGBoost, ultimately identifying AdaBoost as the most effective fraud prediction model. These findings highlight the potential of streamlined feature engineering and ensemble learning techniques for developing efficient and scalable fraud detection systems.

This project aims:

- To identify relevant features to reduce redundancy

- To evaluate performance of machine learning models in predicting fraud by using LightGBM as baseline model

- To determine the best machine learning model in predicting fraud
  
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
- Datasets are not stored in the repository due to GitHub size limits. They will automatically download from GitHub Releases when the app runs for the first time.
- Size:
- Train shape: (300113 rows, 21 columns)
- Test shape: (99887 rows, 21 columns)

**Problem Statement**

Detecting fraudulent transactions is challenging due to class imbalance, temporal drift, and noisy data. Traditional models often fail when minority fraud patterns are overshadowed by the majority of legitimate transactions

**Real World Impact**

- Reduce financial losses due to fraud

- Improve customer trust and retention

- Serve as a deployable ML pipeline integrated with dashboards for operational monitoring

**Key Results**

- 10 relevant features identified from the original feature set, reducing redundancy and model complexity.
  
- Models evaluated: LightGBM (baseline), Random Forest, AdaBoost, and XGBoost.
  
- Best-performing model: AdaBoost.
  
- Highest AUC score achieved: 0.8322.
  
- Reduced feature count by 50% (from 20 to 10 features) while maintaining performance comparable to previous studies.
  
- Demonstrated that feature selection can improve model efficiency without significant loss in predictive capability.

