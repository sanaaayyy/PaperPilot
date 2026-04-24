import os
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from huggingface_hub import hf_hub_download

#CONFIG
OUTPUT_DIR = "artifacts"
MODEL_NAME = "sentence-transformers/allenai-specter"

os.makedirs(OUTPUT_DIR, exist_ok=True)

clf_path = os.path.join(OUTPUT_DIR, "classifier.joblib")
mlb_path = os.path.join(OUTPUT_DIR, "mlb.joblib")
all_papers_path = os.path.join(OUTPUT_DIR, "all_papers.csv")

#LOAD MODELS
encoder = SentenceTransformer(MODEL_NAME)

if not os.path.exists(clf_path) or not os.path.exists(mlb_path):
    st.error("Model files missing in artifacts folder.")
    st.stop()

clf = joblib.load(clf_path)
mlb = joblib.load(mlb_path)

#LOAD EMBEDDINGS FROM HUGGINGFACE
try:
    emb_path = hf_hub_download(
        repo_id="Sanaaayyy/paper-pilot-artifacts",
        filename="paper_embeddings.npy",
        repo_type="dataset"
    )
    paper_embeddings = np.load(emb_path, allow_pickle=True)
except Exception as e:
    st.error(f"Error loading embeddings: {e}")
    st.stop()

#LOAD DATA
if not os.path.exists(all_papers_path):
    st.error("all_papers.csv missing in artifacts folder.")
    st.stop()

try:
    papers_df = pd.read_csv(all_papers_path)
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

#FUNCTIONS
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

    recs = papers_df.iloc[top_idx][
        ["paper_id", "title", "abstract", "categories"]
    ].copy()

    recs["similarity"] = sims[top_idx]
    return recs


#STREAMLIT UI
st.set_page_config(page_title="PaperPilot", layout="wide")

st.title("📚 PaperPilot: Research Paper Recommender")

input_mode = st.radio(
    "Input type",
    ["Title + Abstract", "Short Query / Keywords"]
)

if input_mode == "Title + Abstract":
    title = st.text_input("Paper Title")
    abstract = st.text_area("Paper Abstract")
    query_text = (title.strip() + ". " + abstract.strip()).strip()
else:
    query_text = st.text_area("Enter your research topic / keywords")

top_k = st.slider("Number of recommendations", 3, 20, 5)
threshold = st.slider("Subject prediction threshold", 0.1, 0.9, 0.3, 0.05)

#RUN
if st.button("Analyze") and query_text:
    with st.spinner("Processing..."):
        subjects = predict_subjects(query_text, threshold)
        recs = recommend_similar(query_text, top_k)

    st.subheader("🔎 Predicted Subject Areas")

    if subjects:
        for label, prob in sorted(subjects, key=lambda x: -x[1]):
            st.write(f"{label} — {prob:.3f}")
    else:
        st.write("No subjects above threshold.")

    st.subheader("📄 Recommended Papers")

    for _, row in recs.iterrows():
        st.markdown("---")
        st.markdown(
            f"**{row['title']}**  \n"
            f"`{row['categories']}`  \n"
            f"Similarity: `{row['similarity']:.3f}`"
        )
        st.write(
            row["abstract"][:600] +
            ("..." if len(str(row["abstract"])) > 600 else "")
        )
