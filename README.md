# Intrusion Detection System on NSL‑KDD (Multi‑Class)

This project implements a machine‑learning based Network Intrusion Detection System (NIDS) using the NSL‑KDD benchmark dataset. It includes:

- An offline training pipeline that builds and compares multiple ML models.  
- A Streamlit web application for batch intrusion detection and visualization.  
- A Windows real‑time IDS prototype that captures live traffic and classifies it as Normal or attack (DoS, Probe, R2L, U2R).

***

## 1. Project Overview

The system treats intrusion detection as a supervised multi‑class classification task. NSL‑KDD connection records are preprocessed into numerical feature vectors and used to train several algorithms:

- Random Forest  
- Gradient Boosting  
- XGBoost  
- K‑Nearest Neighbors (KNN)  
- Multi‑Layer Perceptron (MLP)  

All models are trained and evaluated on the standard KDDTrain+ / KDDTest+ split. The best model (by test accuracy) is saved and reused by the Streamlit app and the real‑time IDS script.

***

## 2. Repository Structure

```text
.
├── training_model.py      # Offline training and model selection
├── test_streamlit.py      # Streamlit batch prediction UI
├── realtime_ids.py        # Real-time IDS prototype (Windows)
├── KDDTrain.txt           # NSL-KDD training data (KDDTrain+ style, no header)
├── KDDTest-21.txt         # NSL-KDD test data (KDDTest+ style/subset, no header)
└── artifacts/             # Created after first training run
    ├── encoder.pkl        # OneHotEncoder for categorical features
    ├── scaler.pkl         # StandardScaler for numeric features
    ├── label_encoder.pkl  # LabelEncoder for {normal, dos, probe, r2l, u2r}
    ├── best_model.pkl     # Best model (XGBoost)
    └── other_model.pkl    # Additional saved models (RF, GB, KNN, MLP)
```

***

## 3. Features

- Uses NSL‑KDD KDDTrain+ / KDDTest+ as the data source.  
- Full preprocessing pipeline:
  - Column selection.  
  - One‑hot encoding for categorical features (protocol, service, flag, etc.).  
  - Standardization of numerical features.  
  - Label encoding of target classes (normal, dos, probe, r2l, u2r).  
- Trains and evaluates five supervised learning models with fixed hyperparameters, for example:
  - Random Forest: 200 trees.  
  - Gradient Boosting: 150 estimators.  
  - XGBoost: 150 estimators, multi‑class objective.  
  - KNN: k = 5 neighbors.  
  - MLP: hidden layers (128, 64), max_iter = 300.  
- Selects the best model by test accuracy (typically XGBoost with ≈99.9% accuracy on NSL‑KDD).  
- Streamlit UI:
  - Upload NSL‑KDD‑formatted CSV files.  
  - Choose a saved model.  
  - View sample predictions, per‑class precision/recall/F1, confusion matrix heatmap, Normal vs Attack pie chart, and feature‑importance plot.  
  - Download predictions as a CSV file.  
- Real‑time IDS:
  - Captures packets from a Windows network interface (Npcap/WinPcap).  
  - Aggregates them into NSL‑KDD‑like features.  
  - Applies the trained model and prints live predictions with timestamps.

***

## 4. Requirements

- Python 3.10+ (developed on Python 3.13).  
- Recommended OS for full functionality: Windows 10/11 (for real‑time capture).  

Install the main Python dependencies:

```bash
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn scapy
```

For real‑time packet capture, install Npcap or WinPcap on Windows.

***

## 5. Usage

### 5.1 Offline Training

1. Ensure `KDDTrain.txt` and `KDDTest-21.txt` are in the project folder (or update file paths in `training_model.py`).  
2. Run:

```bash
python training_model.py
```

The script will:

- Load and preprocess the NSL‑KDD training and test data.  
- Train Random Forest, Gradient Boosting, XGBoost, KNN, and MLP models.  
- Print classification reports (precision, recall, F1‑score, support) and accuracies.  
- Choose the best model by test accuracy.  
- Save the best model plus all preprocessing objects (encoder, scaler, label encoder) into `artifacts/`.

### 5.2 Batch Prediction Web App (Streamlit)

After training at least once:

```bash
streamlit run test_streamlit.py
```

In the browser:

1. Select a model from the dropdown (default: `best_model.pkl`).  
2. Upload a CSV file formatted like NSL‑KDD (same column order as KDDTrain+/KDDTest+, no header).  
3. Inspect:
   - Top‑N sample predictions.  
   - Class‑wise metrics.  
   - Confusion matrix heatmap.  
   - Normal vs Attack pie chart.  
   - Feature‑importance visualisation.  
4. Download the predictions as a CSV if desired.

### 5.3 Real‑Time IDS on Windows

1. Open a terminal **as Administrator** to allow packet capture.  
2. Run:

```bash
python realtime_ids.py
```

The script will:

- Select a network interface.  
- Capture packets in short time windows.  
- Transform them into feature vectors using the saved encoder and scaler.  
- Use the best model to classify each window and print output such as:

```text
[YYYY-MM-DD HH:MM:SS] Predicted: normal
```

Terminate the capture with `Ctrl + C`.

***

## 6. Dataset

The system uses NSL‑KDD connection records:

- `KDDTrain.txt` corresponds to KDDTrain+ (training set).  
- `KDDTest-21.txt` corresponds to KDDTest+ or a derived test subset.  

Each record contains 41 original features and a class label. After encoding and scaling, the final representation has about 110 numeric features per record.

***

## 7. Notes and Limitations

- Real‑time evaluation has been done on a single Windows laptop over home Wi‑Fi with mostly benign traffic; no labelled attack traces were injected during live capture.  
- Despite very high overall accuracy, minority classes (R2L and U2R) are still more difficult to detect reliably due to strong class imbalance.  
- The online feature extraction in `realtime_ids.py` approximates the NSL‑KDD connection semantics and demonstrates a proof of concept rather than a production‑ready IDS.

***

## 8. Future Work

- Incorporate newer and more diverse network intrusion datasets, including encrypted traffic.  
- Apply imbalance‑aware learning strategies (class weighting, resampling, focal loss) to improve detection of rare attack types.  
- Package the IDS as a containerized microservice and integrate it with a SIEM or central logging stack for large‑scale deployment and monitoring.

***

## 9. Author

- **Name:** Sagiraju Sai Vikas Varma  
- **Program:** B.Tech–M.Tech CSE (Cybersecurity)  
- **Institution:** School of Cyber Security & Digital Forensics, National Forensic Sciences University  
- **Project Duration:** September 2025 – December 2025