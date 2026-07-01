# 💳 Credit Card Fraud Detection

A machine learning web app that predicts whether a credit card transaction is **fraudulent** or **normal**, with a confidence score. Built with scikit-learn and deployed using Streamlit.

🔗 **Live App:** [Try it here](https://fraud-detection-app-j9imfo2bzrdfvqhtxp7xs9.streamlit.app/)

---

## 📌 Overview

Credit card fraud is rare but costly. This project trains a machine learning model to detect fraudulent transactions from anonymized transaction data, and serves it through an interactive web interface where anyone can test predictions.

## 📊 Dataset

The model is trained on the well-known [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), which contains 284,807 real transactions.

- **Features:** `Time`, `Amount`, and `V1`–`V28`
- **Target:** `Class` (0 = normal, 1 = fraud)
- **Note:** `V1`–`V28` are anonymized using PCA to protect privacy, so their real-world meaning is hidden. The model learns fraud patterns from the numerical relationships between them.

## ⚠️ The Key Challenge: Imbalanced Data

Only about **0.17%** of transactions are fraud. This makes **accuracy misleading** — a model that predicts "normal" for everything would still score ~99.8% accuracy while catching zero fraud.

That's why this project evaluates the model using **precision and recall**, not accuracy:

- **Precision** — of the transactions flagged as fraud, how many were actually fraud (avoiding false alarms)
- **Recall** — of all actual fraud, how much the model caught (avoiding misses)

This distinction is central to real-world fraud detection.

## 🧠 Model

- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Train/test split:** 80/20
- The trained model is saved with `joblib` and loaded by the web app for instant predictions.

## 💻 The Web App

Built with Streamlit, the app lets users:
- Load a real **fraud** or **normal** example with one click
- Enter transaction values manually
- Get a prediction with a **confidence percentage**

## 🚀 Run It Locally

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-app.git
cd fraud-detection-app
pip install -r requirements.txt
streamlit run app.py
