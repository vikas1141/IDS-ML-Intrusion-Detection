import joblib
import time
import pandas as pd
import numpy as np
import os
import warnings
from scapy.all import sniff, IP, TCP, UDP, conf

ARTIFACT_DIR = "artifacts"


encoder = joblib.load(os.path.join(ARTIFACT_DIR, "encoder.pkl"))
scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
meta = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessing_metadata.pkl"))
model = joblib.load(os.path.join(ARTIFACT_DIR, "best_model.pkl"))
label_encoder = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))

WINDOW_SECONDS = 3

def pkt_to_row(pkt):
    feat = {}
    feat["duration"] = 0.0
    if IP in pkt:
        feat["src_bytes"] = len(pkt.payload) if pkt.payload else 0
        feat["dst_bytes"] = len(pkt.payload) if pkt.payload else 0
        proto = pkt[IP].proto
        if proto == 6 and TCP in pkt:
            feat["protocol_type"] = "tcp"
            feat["service"] = str(pkt[TCP].dport)
            feat["flag"] = str(pkt[TCP].flags)
        elif proto == 17 and UDP in pkt:
            feat["protocol_type"] = "udp"
            feat["service"] = str(pkt[UDP].dport)
            feat["flag"] = "SF"
        else:
            feat["protocol_type"] = "other"
            feat["service"] = "other"
            feat["flag"] = "OTH"
    else:
        feat["protocol_type"] = "other"
        feat["service"] = "other"
        feat["flag"] = "OTH"
        feat["src_bytes"] = 0
        feat["dst_bytes"] = 0

    row = {}
    for c in meta["numeric_cols"]:
        row[str(c)] = 0.0
    if len(meta["numeric_cols"]) >= 2:
        row[str(meta["numeric_cols"][0])] = float(feat.get("src_bytes", 0))
        row[str(meta["numeric_cols"][1])] = float(feat.get("dst_bytes", 0))
    row.update({
        str(meta["categorical_cols"][0]): feat["protocol_type"],
        str(meta["categorical_cols"][1]): feat["service"],
        str(meta["categorical_cols"][2]): feat["flag"]
    })
    return row

from scapy.all import get_if_list


def capture_window(timeout=WINDOW_SECONDS, iface_selected_holder={"iface": None}):
    """
    Auto-detect an interface that captures packets, then sniff on that interface.
    iface_selected_holder is a dict used to persist chosen interface across calls.
    """
   
    chosen_iface = iface_selected_holder.get("iface")
    if chosen_iface:
        try:
            pkts = sniff(iface=chosen_iface, filter="ip", timeout=timeout, store=1)
            return [pkt_to_row(p) for p in pkts if IP in p]
        except Exception as e:
            
            iface_selected_holder["iface"] = None
            chosen_iface = None

   
    ifaces = get_if_list()
    for iface in ifaces:
        try:
            pkts = sniff(iface=iface, filter="ip", timeout=0.8, store=1)
            if pkts and len(pkts) > 0:
                
                iface_selected_holder["iface"] = iface
                print(f"Auto-selected interface for capture: {iface} (captured {len(pkts)} packets during probe)")
                return [pkt_to_row(p) for p in pkts if IP in p]
        except Exception:
         
            continue

    
    try:
        pkts = sniff(filter="ip", timeout=timeout, store=1)
        return [pkt_to_row(p) for p in pkts if IP in p]
    except Exception as e:
      
        return []


def preprocess_and_predict(df_rows):
    if not df_rows:
        return None
    df = pd.DataFrame(df_rows)
    Xraw = pd.DataFrame()
    for c in meta["numeric_cols"]:
        key = str(c)
        if key in df.columns:
            Xraw[c] = df[key]
        else:
            Xraw[c] = 0.0
    for i, c in enumerate(meta["categorical_cols"]):
        key = str(c)
        if key in df.columns:
            Xraw[c] = df[key]
        else:
            Xraw[c] = "other"

   
    X_num = scaler.transform(Xraw[meta["numeric_cols"]].astype(float))
    X_num_df = pd.DataFrame(X_num, columns=[str(c) for c in meta["numeric_cols"]])
   
    cat_arr = encoder.transform(Xraw[meta["categorical_cols"]].astype(str))
    try:
        cat_arr = cat_arr.toarray()
    except Exception:
        pass
    try:
        cat_cols = encoder.get_feature_names_out([str(c) for c in meta["categorical_cols"]])
    except Exception:
       
        cat_cols = []
        for i, c in enumerate(meta["categorical_cols"]):
            cats = getattr(encoder, "categories_", [])[i] if hasattr(encoder, "categories_") else []
            if len(cats) > 0:
                cat_cols.extend([f"{c}_{cat}" for cat in cats])
    X_cat_df = pd.DataFrame(cat_arr, columns=cat_cols)
    X_final = pd.concat([X_num_df, X_cat_df], axis=1)
    X_final = X_final.reindex(columns=X_final.columns, fill_value=0)

    preds = model.predict(X_final)
    try:
        return label_encoder.inverse_transform(preds)[0]
    except Exception:
        return preds[0]

if __name__ == "__main__":
    print("Starting real-time IDS capture (Windows L3 mode). Run as Administrator. Press Ctrl+C to stop.")
    try:
        while True:
            rows = capture_window()
            if not rows:
                print(f"[{time.strftime('%H:%M:%S')}] No packets in window.")
                time.sleep(1)
                continue
            pred = preprocess_and_predict(rows)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Predicted: {pred}")
    except KeyboardInterrupt:
        print("Stopping capture.")
