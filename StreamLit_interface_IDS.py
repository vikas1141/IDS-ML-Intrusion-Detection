import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="AI IDS - NSL-KDD (Multi-class)", layout="wide")
metric_css = """
<style>
[data-testid="metric-container"] {
    background: #ffffff10;
    border: 1px solid #7fffd4;
    border-radius: 10px;
    padding: 10px;
}
[data-testid="metric-container"] > div:nth-child(1) > div {
    color: black !important;
    font-size: 15px !important;
}
[data-testid="metric-container"] > div:nth-child(2) > div {
    color: black !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}
</style>
"""
st.markdown(metric_css, unsafe_allow_html=True)

ARTIFACT_DIR = "artifacts"
EXPECTED_ARTIFACTS = ["encoder.pkl", "scaler.pkl", "preprocessing_metadata.pkl", "label_encoder.pkl"]


@st.cache_resource
def load_preprocessors():
    missing = [p for p in EXPECTED_ARTIFACTS if not os.path.exists(os.path.join(ARTIFACT_DIR, p))]
    if missing:
        raise FileNotFoundError(f"Missing artifacts in '{ARTIFACT_DIR}': {missing}. Run train_model.py first.")
    encoder = joblib.load(os.path.join(ARTIFACT_DIR, "encoder.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    metadata = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessing_metadata.pkl"))
    label_encoder = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))
    return encoder, scaler, metadata, label_encoder

def detect_label_column(df):
    for c in [df.shape[1]-1, df.shape[1]-2, df.shape[1]-3]:
        if c >= 0:
            col_vals = df[c].astype(str).str.lower()
            if col_vals.str.contains("normal|neptune|smurf|back|satan|ipsweep|portsweep|warez|guess_passwd", regex=True).sum() > 0:
                return c
    return None

def transform_input(X_raw, encoder, scaler, categorical_cols, numeric_cols, cat_feature_names):
    
    num_df = pd.DataFrame()
    cat_df = pd.DataFrame()
    present_num = [c for c in numeric_cols if c in X_raw.columns]
    if present_num:
        try:
            num_df = pd.DataFrame(scaler.transform(X_raw[present_num].astype(float)), columns=[str(c) for c in present_num], index=X_raw.index)
        except Exception as e:
            raise ValueError(f"Numeric scaling failed: {e}")
    present_cat = [c for c in categorical_cols if c in X_raw.columns]
    if present_cat:
        try:
            cat_arr = encoder.transform(X_raw[present_cat].astype(str))
           
            try:
                cat_arr = cat_arr.toarray()
            except Exception:
                pass
            cat_cols = encoder.get_feature_names_out([str(c) for c in present_cat])
            cat_df = pd.DataFrame(cat_arr, columns=cat_cols, index=X_raw.index)
        except Exception as e:
            raise ValueError(f"Categorical encoding failed: {e}")
    X_final = pd.concat([num_df, cat_df], axis=1)
    expected_cols = [str(c) for c in numeric_cols] + [str(x) for x in cat_feature_names]
    X_final = X_final.reindex(columns=expected_cols, fill_value=0)
    return X_final


st.title("Intrusion Detection System (NSL-KDD) — Multi-class")
st.write("Upload a KDD-formatted file (no header, comma-separated). App will load models on demand (fast UI).")


try:
    encoder, scaler, metadata, label_encoder = load_preprocessors()
except Exception as e:
    st.error(f"Preprocessors not found or could not be loaded: {e}")
    st.stop()

categorical_cols = metadata.get("categorical_cols", [])
numeric_cols = metadata.get("numeric_cols", [])
cat_feature_names = metadata.get("cat_feature_names", [])


model_files = [f for f in os.listdir(ARTIFACT_DIR) if f.endswith("_model.pkl") or f=="best_model.pkl"]
model_choice = None
if model_files:
    model_choice = st.sidebar.selectbox("Choose saved model (loads on demand)", options=model_files)
else:
    st.sidebar.warning("No model files found in artifacts/. Run training script first.")

st.sidebar.checkbox("Show feature importance (if available)", value=True, key="feat_imp")
st.sidebar.checkbox("Enable prediction download", value=True, key="download_enable")
st.sidebar.markdown("---")
st.sidebar.info("For live capture use realtime_ids.py in a separate terminal (requires root/Admin).")

uploaded = st.file_uploader("Upload NSL-KDD CSV / TXT file", type=["csv","txt"])
if not uploaded:
    st.info("Upload a dataset (KDDTrain+/KDDTest+ format) to run predictions.")
    st.stop()


try:
    df = pd.read_csv(uploaded, header=None)
    st.write(f"Uploaded file shape: {df.shape}")
except Exception as e:
    st.error(f"Failed to read uploaded file: {e}")
    st.stop()


label_col = detect_label_column(df)
if label_col is not None:
    st.info(f"Detected label column at index {label_col}. Will use for evaluation.")
    X_raw = df.drop(columns=list(range(label_col, df.shape[1])))
    y_true_raw = df[label_col].astype(str)
else:
    X_raw = df.copy()
    y_true_raw = None

st.write(f"Expected categorical cols: {categorical_cols}")
st.write(f"Expected numeric cols: first 6 -> {numeric_cols[:6]} ... total {len(numeric_cols)}")

try:
    X_final = transform_input(X_raw, encoder, scaler, categorical_cols, numeric_cols, cat_feature_names)
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()


if model_choice is None:
    st.warning("No model selected. Select a model from the sidebar.")
    st.stop()

model_path = os.path.join(ARTIFACT_DIR, model_choice)
try:
    st.info(f"Loading model: {model_choice}  (this may take a few seconds)")
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Failed to load model '{model_choice}': {e}")
    st.stop()


try:
    preds = model.predict(X_final)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()


try:
    if hasattr(label_encoder, "inverse_transform") and np.issubdtype(preds.dtype, np.number):
        preds_labels = label_encoder.inverse_transform(preds)
    else:
        preds_labels = preds.astype(str)
except Exception:
    preds_labels = preds.astype(str)

result_df = X_raw.copy()
result_df["prediction"] = preds_labels


total = len(preds_labels)
normal_count = sum(1 for p in preds_labels if str(p).strip().lower()=="normal")
attack_count = total - normal_count
attack_rate = attack_count/total*100

st.subheader("Summary Metrics")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total", total)
c2.metric("Normal", normal_count)
c3.metric("Attack", attack_count)
c4.metric("Attack Rate (%)", f"{attack_rate:.2f}%")

if attack_rate > 50:
    st.error(f"High Risk — attack traffic {attack_rate:.2f}%")
elif attack_rate > 30:
    st.warning(f"Warning — attack traffic {attack_rate:.2f}%")
else:
    st.success(f"Network appears stable (attack rate {attack_rate:.2f}%)")

fig = px.pie(values=[normal_count, attack_count], names=["Normal","Attack"], title="Normal vs Attack")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Sample predictions (top 30)")
st.dataframe(result_df.head(30))


if y_true_raw is not None:
    def map_label_series(s):
        s = s.astype(str).str.strip().str.lower()
        def mapper(x):
            if "normal" in x: return "normal"
            if any(k in x for k in ["neptune","smurf","back","teardrop","pod","land"]): return "dos"
            if any(k in x for k in ["satan","ipsweep","nmap","portsweep","mscan"]): return "probe"
            if any(k in x for k in ["warez","ftp","guess","imap","phf","sendmail"]): return "r2l"
            if any(k in x for k in ["overflow","rootkit","loadmodule","perl"]): return "u2r"
            return "other"
        return s.apply(mapper)
    y_true_mapped = map_label_series(y_true_raw)
    y_pred_mapped = pd.Series(preds_labels).astype(str).str.strip().str.lower().apply(lambda x: x if x in ["normal","dos","probe","r2l","u2r","other"] else "other")

    st.subheader("Evaluation (if ground-truth present)")
    try:
        cr = classification_report(y_true_mapped, y_pred_mapped, output_dict=True, zero_division=0)
        cr_df = pd.DataFrame(cr).transpose()
        st.dataframe(cr_df)
    except Exception as e:
        st.warning(f"Could not show classification report: {e}")

    labels = ["normal","dos","probe","r2l","u2r","other"]
    cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=labels)
    fig_cm, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig_cm)


if st.sidebar.checkbox("Show feature importance (if available)", value=True):
    st.subheader("Feature importance (if available)")
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        feats = X_final.columns
        fi_df = pd.DataFrame({"feature": feats, "importance": fi}).sort_values("importance", ascending=False).head(40)
        fig = px.bar(fi_df, x="importance", y="feature", orientation="h", title="Top feature importances")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Selected model does not expose feature_importances_ (e.g., MLP, KNN).")


if st.sidebar.checkbox("Enable prediction download", value=True):
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.write("Model files present in artifacts/:")
st.write(model_files)
st.caption("Note: For live capture, run realtime_ids.py separately (requires root/Admin).")
