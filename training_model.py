# train_model.py  (corrected - uses LabelEncoder for training & evaluation)
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

# ----------------- CONFIG -----------------
DATA_PATH = "KDDTrain+.txt"   # change if your file has a different name
ARTIFACT_DIR = "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.2
Path(ARTIFACT_DIR).mkdir(exist_ok=True)
# ------------------------------------------

# Explicit schema based on your sample rows:
# features are columns 0..40 (41 feature columns)
LABEL_COL_INDEX = 41  # 0-based index of label (second last column)
DIFFICULTY_COL_INDEX = 42

# Categorical and numeric columns (0-based indices)
CATEGORICAL_COLS = [1, 2, 3]  # protocol_type, service, flag
NUMERIC_COLS = [c for c in range(0, 41) if c not in CATEGORICAL_COLS]

# mapping raw attack names to 5 classes
def map_attack_to_category(lbl):
    s = str(lbl).strip().lower().rstrip('.')
    if "normal" in s:
        return "normal"
    # dos
    dos_keys = ["neptune","smurf","back","teardrop","pod","land","apache2","processtable","udpstorm","mailbomb","worm"]
    if any(k in s for k in dos_keys):
        return "dos"
    # probe
    probe_keys = ["satan","ipsweep","nmap","portsweep","mscan","saint"]
    if any(k in s for k in probe_keys):
        return "probe"
    # r2l
    r2l_keys = ["ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster","sendmail"]
    if any(k in s for k in r2l_keys):
        return "r2l"
    # u2r
    u2r_keys = ["buffer_overflow","loadmodule","rootkit","perl","xterm"]
    if any(k in s for k in u2r_keys):
        return "u2r"
    return "other"

def load_and_split(path):
    df = pd.read_csv(path, header=None)
    if df.shape[1] <= LABEL_COL_INDEX:
        raise ValueError(f"File {path} appears to have too few columns ({df.shape[1]}). Expecting label at index {LABEL_COL_INDEX}.")
    X = df.drop(columns=[LABEL_COL_INDEX, DIFFICULTY_COL_INDEX])
    y_raw = df[LABEL_COL_INDEX].astype(str)
    y = y_raw.apply(map_attack_to_category)
    # keep only relevant classes (we will keep 'other' too)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)

def fit_preprocessors(X_train):
    # Use OneHotEncoder handle_unknown=ignore. Older sklearns default to sparse output; convert to dense as needed
    encoder = OneHotEncoder(handle_unknown="ignore")
    scaler = StandardScaler()
    if len(CATEGORICAL_COLS) > 0:
        encoder.fit(X_train[CATEGORICAL_COLS].astype(str))
        try:
            cat_feature_names = encoder.get_feature_names_out([str(c) for c in CATEGORICAL_COLS])
        except Exception:
            # fallback name generation
            cat_feature_names = []
            for i, c in enumerate(CATEGORICAL_COLS):
                cats = encoder.categories_[i]
                cat_feature_names.extend([f"{c}_{cat}" for cat in cats])
    else:
        cat_feature_names = []
    if len(NUMERIC_COLS) > 0:
        scaler.fit(X_train[NUMERIC_COLS].astype(float))
    metadata = {
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "cat_feature_names": [str(x) for x in cat_feature_names]
    }
    return encoder, scaler, metadata

def transform_df(X, encoder, scaler):
    parts = []
    if len(NUMERIC_COLS) > 0:
        num_arr = scaler.transform(X[NUMERIC_COLS].astype(float))
        num_df = pd.DataFrame(num_arr, columns=[str(c) for c in NUMERIC_COLS], index=X.index)
        parts.append(num_df)
    if len(CATEGORICAL_COLS) > 0:
        cat_arr = encoder.transform(X[CATEGORICAL_COLS].astype(str))
        # cat_arr may be sparse matrix depending on sklearn version -> convert to dense
        try:
            cat_arr = cat_arr.toarray()
        except Exception:
            pass
        try:
            cat_cols = encoder.get_feature_names_out([str(c) for c in CATEGORICAL_COLS])
        except Exception:
            # fallback columns
            cat_cols = []
            for i, c in enumerate(CATEGORICAL_COLS):
                cats = encoder.categories_[i]
                cat_cols.extend([f"{c}_{cat}" for cat in cats])
        cat_df = pd.DataFrame(cat_arr, columns=cat_cols, index=X.index)
        parts.append(cat_df)
    if parts:
        X_final = pd.concat(parts, axis=1)
    else:
        X_final = pd.DataFrame(index=X.index)  # empty
    X_final.columns = X_final.columns.astype(str)
    return X_final

def train_models(X_train, y_train_enc, X_test, y_test_enc, label_encoder):
    # All models trained on encoded numeric labels (y_*_enc)
    models = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, random_state=RANDOM_STATE),
        "xgboost": xgb.XGBClassifier(n_estimators=150, use_label_encoder=False, eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "mlp": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=RANDOM_STATE)
    }
    results = {}
    for name, clf in models.items():
        print(f"\nTraining {name} ...")
        clf.fit(X_train, y_train_enc)
        preds_enc = clf.predict(X_test)
        # inverse transform to string labels for reporting
        try:
            preds = label_encoder.inverse_transform(preds_enc)
            y_test_str = label_encoder.inverse_transform(y_test_enc)
        except Exception:
            # fallback: if label_encoder not available, map integers to strings by str()
            preds = np.array([str(p) for p in preds_enc])
            y_test_str = np.array([str(p) for p in y_test_enc])
        acc = accuracy_score(y_test_enc, preds_enc)
        print(f"{name} accuracy: {acc:.4f}")
        print(classification_report(y_test_str, preds, zero_division=0))
        joblib.dump(clf, os.path.join(ARTIFACT_DIR, f"{name}_model.pkl"))
        results[name] = {"model": clf, "accuracy": acc}
    # Save best by accuracy
    best_name = max(results.items(), key=lambda kv: kv[1]["accuracy"])[0]
    joblib.dump(results[best_name]["model"], os.path.join(ARTIFACT_DIR, "best_model.pkl"))
    return results

def main():
    print("Loading and splitting dataset...")
    X_train_raw, X_test_raw, y_train, y_test = load_and_split(DATA_PATH)
    print("Fitting preprocessors...")
    encoder, scaler, metadata = fit_preprocessors(X_train_raw)
    print("Transforming data...")
    X_train = transform_df(X_train_raw, encoder, scaler)
    X_test = transform_df(X_test_raw, encoder, scaler)
    # align test to train columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    # Save preprocessors and metadata
    joblib.dump(encoder, os.path.join(ARTIFACT_DIR, "encoder.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    joblib.dump(metadata, os.path.join(ARTIFACT_DIR, "preprocessing_metadata.pkl"))
    print("Saved encoder/scaler/metadata.")
    # Label encoding (important: fit on combined train+test to ensure consistent mapping)
    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_test], axis=0))
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    joblib.dump(le, os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))
    print("Saved label_encoder.pkl")
    # Train models (all on encoded labels)
    results = train_models(X_train, y_train_enc, X_test, y_test_enc, le)
    print("Training complete. Models saved in artifacts/")
    # print summary
    for name, info in results.items():
        print(f"{name}: acc={info['accuracy']:.4f}")
    best_name = max(results.items(), key=lambda kv: kv[1]["accuracy"])[0]
    print(f"Best model is: {best_name}")

main()
