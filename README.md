# Credit Risk Prediction

## Project Overview
This project aims to predict the likelihood of a customer defaulting on a loan using an Artificial Neural Network (ANN).

The application provides an interactive interface built with Streamlit to explore customer data, generate credit risk predictions, and evaluate model performance.

## Business Problem
Financial institutions need to assess the potential risk associated with lending money to customers.

This project uses historical customer financial information to help identify customers who may have a higher likelihood of default.

## Dataset
The dataset contains customer financial and credit-related features.

- Target Variable: `SeriousDlqin2yrs`
- 0: Non-Default
- 1: Default
- Number of Features: 10

## Technologies & Libraries
- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Plotly
- Streamlit

## Machine Learning Model
An Artificial Neural Network (ANN) was developed for binary classification.

The model predicts the probability of customer default based on financial and credit-related features.

## Application Features

### 1. Exploratory Data Analysis (EDA)
- Data Overview
- Univariate Analysis
- Bivariate Analysis
- Multivariate Analysis

### 2. Credit Risk Prediction
- Customer risk probability
- Adjustable classification threshold
- Risk classification

### 3. Model Performance
- Data Overview
- Confusion Matrix
- Precision-Recall Curve
- Threshold Analysis
- Performance Metrics

## Project Structure
```text
credit-risk-prediction/
│
├── app.py
├── credit_risk_model.keras
├── scaler_pipeline.pkl
├── credit_risk_test.csv
├── requirements.txt
└── README.md
```

## How to Run Locally

1. Clone the repository.
2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run app.py
```

## Disclaimer
This project is intended for educational and analytical purposes. Its predictions should support human decision-making and should not be used as the sole basis for credit approval or rejection.