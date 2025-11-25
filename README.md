Intrusion Detection System using Machine Learning (IDS-ML)

Overview:
This project implements a Machine Learning–based Intrusion Detection System (IDS) that identifies malicious network traffic. It analyzes network connection records and classifies them as Normal or Attack using the NSL-KDD dataset. The system improves detection compared to traditional rule-based IDS by learning attack behavior patterns from real data. A Streamlit web dashboard is included to upload datasets, perform real-time predictions, display attack statistics, visualize charts, and download results.

Features:

Binary classification: Normal vs Attack

Data preprocessing with One-Hot Encoding and Standard Scaling

SMOTE oversampling for class imbalance

Random Forest classifier achieving 99.92% accuracy

Real-time Streamlit dashboard with charts and metrics

Supports CSV download of predicted results

Tech Stack:
Python, Scikit-Learn, Streamlit, Plotly, Joblib, NSL-KDD Dataset

Model Performance:
Accuracy: 99.92%
False Positives: 6
False Negatives: 14
Attack Recall: 99.9%
Confusion Matrix:
[[11712, 14],
[6, 13463]]

How to Run:

Install dependencies:
pip install streamlit pandas numpy scikit-learn imbalanced-learn joblib plotly

Launch the web app:
python -m streamlit run app.py

Upload dataset file (e.g., KDDTest+.txt) to view predictions

Future Enhancements:

Multi-class attack classification

Deep learning (LSTM/CNN)

Real-time packet capture with Scapy

SOC / SIEM integration support

Author:
Sagiraju Sai Vikas Varma
B.Tech CSE (Cybersecurity)

Project Status:
Active – open for enhancements and contributions
