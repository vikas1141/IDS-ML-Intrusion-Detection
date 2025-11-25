

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


metric_css = """
<style>
[data-testid="metric-container"] {
    background: #ffffff10;
    border: 1px solid #7fffd4;
    border-radius: 10px;
    padding: 10px;
}
[data-testid="metric-container"] > div:nth-child(1) > div {
    color: black !important;   /* title color */
    font-size: 15px !important;
}
[data-testid="metric-container"] > div:nth-child(2) > div {
    color: black !important;   /* value text color */
    font-size: 28px !important;
    font-weight: 800 !important;
}
</style>
"""
st.markdown(metric_css, unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("random_forest_model.pkl")
    metadata = joblib.load("preprocessing_metadata.pkl")

    categorical_cols = metadata["categorical_cols"]
    numeric_cols = metadata["numeric_cols"]
    cat_feature_names = metadata["cat_feature_names"]

    return encoder, scaler, model, categorical_cols, numeric_cols, cat_feature_names


encoder, scaler, model, categorical_cols, numeric_cols, cat_feature_names = load_artifacts()

st.title("Intrusion Detection System (NSL-KDD) using Machine Learning")
st.write(
    "Upload network connection records in NSL-KDD format and the model will classify them as "
    "**normal** or **attack**."
)

st.markdown("""
**Input format note**  
- Same format as `KDDTrain+.txt` / `KDDTest+.txt`  
- 43 columns, comma-separated, no header  
- Column 41 = label (optional, will be ignored if present)  
- Column 42 = difficulty level (will be ignored)
""")

uploaded_file = st.file_uploader("Upload NSL-KDD CSV file", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        
        df = pd.read_csv(uploaded_file, header=None)
        st.write(f"File loaded with shape: {df.shape}")

        
        if df.shape[1] >= 43:
            label_col = 41
            difficulty_col = 42
            X_raw = df.drop(columns=[label_col, difficulty_col])
        else:
            X_raw = df.copy()

       
        missing_cats = [c for c in categorical_cols if c not in X_raw.columns]
        if missing_cats:
            st.error(f"Input file missing expected categorical columns: {missing_cats}")
        else:
            cat_data = X_raw[categorical_cols].astype(str)
            encoded_cat = encoder.transform(cat_data)
            cat_df = pd.DataFrame(encoded_cat, columns=cat_feature_names, index=X_raw.index)

        
            num_cols_present = [c for c in numeric_cols if c in X_raw.columns]
            num_data = X_raw[num_cols_present].astype(float)
            scaled_num = scaler.transform(num_data)
            num_df = pd.DataFrame(scaled_num, columns=num_cols_present, index=X_raw.index)

          
            X_final = pd.concat([num_df, cat_df], axis=1)
            X_final.columns = X_final.columns.astype(str)

         

            preds = model.predict(X_final)
            result_df = df.copy()
            result_df["prediction"] = preds
            pred_counts = pd.Series(preds).value_counts()

          
            total_records = len(preds)
            attack_count = pred_counts.get("attack", 0)
            normal_count = pred_counts.get("normal", 0)
            attack_rate = (attack_count / total_records) * 100

            st.subheader("Network Intrusion Detection Dashboard")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Records", total_records)

            with col2:
                st.metric("Normal Traffic", normal_count)

            with col3:
                st.metric("Attack Traffic", attack_count)

            with col4:
                st.metric("Attack Rate (%)", f"{attack_rate:.2f}%")



         
            if attack_rate > 50:
                st.error(f"High Risk: Attack traffic is {attack_rate:.2f}% of total connections!")
            elif attack_rate > 30:
                st.warning(f"Warning: Attack traffic is {attack_rate:.2f}% of total connections.")
            else:
                st.success(f"Status: Network appears stable (Attack rate {attack_rate:.2f}%).")


        
            fig = px.pie(values=[normal_count, attack_count], names=["Normal", "Attack"], title="Attack vs Normal Distribution")
            st.plotly_chart(fig)

           
            st.subheader("Sample Predictions (Top 20)")
            st.dataframe(result_df.head(20))

    except Exception as e:
        st.error(f"Error while processing file: {e}")
else:
    st.info("Please upload a NSL-KDD formatted file to begin.")
