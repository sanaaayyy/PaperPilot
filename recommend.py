import os
import numpy as np
import pandas as pd
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["predict", "recommend"], required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "predict":
        subjects = predict_subjects(args.text)
        print("Predicted subjects:")
        for label, prob in subjects:
            print(f"{label}: {prob:.3f}")
    else:
        recs = recommend_similar(args.text, top_k=args.top_k)
        for _, row in recs.iterrows():
            print("=" * 80)
            print(f"ID: {row['paper_id']}")
            print(f"Title: {row['title']}")
            print(f"Categories: {row['categories']}")
            print(f"Similarity: {row['similarity']:.3f}")
            print(f"Abstract: {row['abstract'][:500]}...")