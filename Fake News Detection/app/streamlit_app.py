import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import io
import pandas as pd
import streamlit as st

from src.utils import clean_text
from src.predict_tfidf import load_pipeline

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection (ML + Optional DistilBERT)")
st.caption("Choose LR / SVM / DistilBERT. Output label: 1=REAL, 0=FAKE")

MODELS = {
    "TF‑IDF + Logistic Regression": ("tfidf", "models/pipeline_lr.joblib"),
    "TF‑IDF + Linear SVM": ("tfidf", "models/pipeline_svm.joblib"),
    "DistilBERT (fine‑tuned)": ("bert", "models/bert_distilbert"),
}

def model_exists(kind, path):
    if kind == "tfidf":
        return os.path.exists(path)
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))

@st.cache_resource
def load_any_model(kind, path):
    if kind == "tfidf":
        return load_pipeline(path)
    from src.predict_bert import BertPredictor
    return BertPredictor(path)

st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Choose model", list(MODELS.keys()))
kind, path = MODELS[model_name]

if not model_exists(kind, path):
    st.warning(
        f"Model not found: `{path}`\n\n"
        "Train it first:\n"
        "- Prepare data: `python -m src.prepare_data --input data/raw.csv --output data/cleaned.csv`\n"
        "- LR: `python -m src.train_tfidf --data data/cleaned.csv --model lr`\n"
        "- SVM: `python -m src.train_tfidf --data data/cleaned.csv --model svm`\n"
        "- BERT: `python -m src.train_bert --data data/cleaned.csv --out models/bert_distilbert`"
    )
    st.stop()

model = load_any_model(kind, path)

tab1, tab2 = st.tabs(["Single Text", "CSV Batch Upload"])

with tab1:
    st.subheader("Single article prediction")
    text = st.text_area("Paste news text here:", height=200, placeholder="Enter a news article or claim...")
    if st.button("Predict", type="primary"):
        if not text.strip():
            st.error("Please enter some text.")
        else:
            cleaned = clean_text(text)
            if kind == "tfidf":
                label = int(model.predict([cleaned])[0])
                clf = model.named_steps["clf"]
                Xv = model.named_steps["tfidf"].transform([cleaned])
                conf = None
                if hasattr(clf, "predict_proba"):
                    conf = float(clf.predict_proba(Xv)[0][1])
                elif hasattr(clf, "decision_function"):
                    import math
                    conf = 1.0 / (1.0 + math.exp(-float(clf.decision_function(Xv)[0])))
            else:
                r = model.predict(cleaned)
                label = int(r["label"])
                conf = r.get("confidence_real", None)

            st.success(f"Prediction: **{'REAL' if label==1 else 'FAKE'}**")
            if conf is not None:
                st.write(f"Confidence (REAL): **{conf:.3f}**")

with tab2:
    st.subheader("Batch prediction via CSV")
    st.write("Upload a CSV that contains a `text` column (optionally `title`).")
    up = st.file_uploader("Upload CSV", type=["csv"])

    if up is not None:
        df = pd.read_csv(up)

        if "text" not in df.columns:
            st.error(f"CSV must contain a `text` column. Found: {list(df.columns)}")
            st.stop()

        if "title" in df.columns:
            merged = (df["title"].fillna("").astype(str) + ". " + df["text"].fillna("").astype(str)).str.strip()
        else:
            merged = df["text"].fillna("").astype(str)

        cleaned_texts = merged.map(clean_text)

        if st.button("Run batch prediction", type="primary"):
            if kind == "tfidf":
                labels = model.predict(cleaned_texts.tolist())
                clf = model.named_steps["clf"]
                Xv = model.named_steps["tfidf"].transform(cleaned_texts.tolist())

                confs = [None] * len(labels)
                if hasattr(clf, "predict_proba"):
                    confs = clf.predict_proba(Xv)[:, 1].tolist()
                elif hasattr(clf, "decision_function"):
                    import numpy as np
                    margins = clf.decision_function(Xv)
                    confs = (1 / (1 + np.exp(-margins))).tolist()
            else:
                labels, confs = [], []
                for t in cleaned_texts.tolist():
                    r = model.predict(t)
                    labels.append(r["label"])
                    confs.append(r.get("confidence_real", None))

            out = df.copy()
            out["pred_label"] = [int(x) for x in labels]
            out["pred_label_name"] = ["REAL" if int(x) == 1 else "FAKE" for x in labels]
            out["confidence_real"] = confs

            st.dataframe(out.head(30), use_container_width=True)

            buf = io.BytesIO()
            out.to_csv(buf, index=False)
            st.download_button(
                "Download predictions CSV",
                data=buf.getvalue(),
                file_name="predictions.csv",
                mime="text/csv"
            )
