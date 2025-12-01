import os
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

OUTPUT_DIR = "artifacts"
MODEL_NAME = "sentence-transformers/allenai-specter"

clf_path = os.path.join(OUTPUT_DIR, "classifier.joblib")
mlb_path = os.path.join(OUTPUT_DIR, "mlb.joblib")
emb_path = os.path.join(OUTPUT_DIR, "paper_embeddings.npy")
all_papers_path = os.path.join(OUTPUT_DIR, "all_papers.csv")

encoder = SentenceTransformer(MODEL_NAME)
clf = joblib.load(clf_path)
mlb = joblib.load(mlb_path)
import urllib.request

EMB_URL = " https://huggingface.co/datasets/Sanaaayyy/paper-pilot-artifacts/tree/main"
emb_path = os.path.join(OUTPUT_DIR, "paper_embeddings.npy")

if not os.path.exists(emb_path):
    urllib.request.urlretrieve(EMB_URL, emb_path)

paper_embeddings = np.load(emb_path)
papers_df = pd.read_csv(all_papers_path)

def encode_query(text):
    emb = encoder.encode(
        [text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return emb[0]

def predict_subjects(text, threshold=0.3):
    emb = encode_query(text).reshape(1, -1)
    probs = clf.predict_proba(emb)[0]
    indices = np.where(probs >= threshold)[0]
    labels = mlb.classes_[indices]
    return list(zip(labels, probs[indices]))

def recommend_similar(text, top_k=10):
    query_emb = encode_query(text).reshape(1, -1)
    sims = cosine_similarity(query_emb, paper_embeddings)[0]
    top_idx = np.argsort(-sims)[:top_k]
    recs = papers_df.iloc[top_idx][["paper_id", "title", "abstract", "categories"]].copy()
    recs["similarity"] = sims[top_idx]
    return recs

st.set_page_config(page_title="Research Paper Recommender", layout="wide")

st.title("Research Paper Recommendation and Subject Area Prediction")

input_mode = st.radio("Input type", ["Title + Abstract", "Short Query / Keywords"])

if input_mode == "Title + Abstract":
    title = st.text_input("Paper Title")
    abstract = st.text_area("Paper Abstract")
    query_text = (title.strip() + ". " + abstract.strip()).strip()
else:
    query_text = st.text_area("Enter your research topic / keywords")

top_k = st.slider("Number of recommendations", min_value=3, max_value=20, value=5)
threshold = st.slider("Subject prediction threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.05)

if st.button("Analyze") and query_text:
    with st.spinner("Computing predictions..."):
        subjects = predict_subjects(query_text, threshold=threshold)
        recs = recommend_similar(query_text, top_k=top_k)

    st.subheader("Predicted Subject Areas")
    if subjects:
        for label, prob in sorted(subjects, key=lambda x: -x[1]):
            st.write(f"{label} – {prob:.3f}")
    else:
        st.write("No subjects above threshold.")

    st.subheader("Recommended Papers")
    for _, row in recs.iterrows():
        st.markdown("---")
        st.markdown(f"**{row['title']}**  \n`{row['categories']}`  \nSimilarity: `{row['similarity']:.3f}`")
        st.write(row["abstract"][:600] + ("..." if len(str(row["abstract"])) > 600 else ""))
        
       