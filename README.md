# PaperPilot: Research Paper Recommendation and Subject Prediction System

## Overview

PaperPilot is an intelligent research assistant designed to help users explore academic literature more efficiently. The system takes a research idea or query as input and performs two key tasks:

* Predicts relevant subject areas
* Recommends similar research papers

It uses transformer-based embeddings to understand the semantic meaning of text rather than relying on simple keyword matching.

---

## Key Features

* **Semantic Representation of Text**
  Uses the SPECTER model to generate embeddings specifically suited for scientific documents.

* **Multi-label Subject Classification**
  Predicts multiple subject categories using a trained One-vs-Rest classifier.

* **Similarity-based Recommendation**
  Retrieves relevant papers using cosine similarity over precomputed embeddings.

* **Adjustable Prediction Threshold**
  Allows control over how confident predictions must be before being displayed.

* **Scalable Design**
  Large embedding files are hosted externally and loaded dynamically during runtime.

---

## System Architecture

```
User Input (Query / Title + Abstract)
                │
                ▼
     SentenceTransformer (SPECTER)
                │
                ▼
     Query Embedding Vector
                │
        ┌───────┴────────┐
        ▼                ▼
Classification      Similarity Search
(Logistic Reg.)     (Cosine Similarity)
        ▼                ▼
Predicted Labels   Recommended Papers
```

---

## Technologies Used

* Python
* Streamlit
* Sentence Transformers (allenai-specter)
* Scikit-learn
* Pandas, NumPy
* Hugging Face Hub

---

## Project Structure

```
PaperPilot/
│
├── app.py                 # Streamlit application
├── requirements.txt       # Dependencies
│
├── artifacts/
│   ├── classifier.joblib
│   ├── mlb.joblib
│   ├── all_papers.csv
│
└── README.md
```

---

## How It Works

1. **Input Processing**
   The user provides either a title and abstract or a short query.

2. **Text Encoding**
   The input is converted into a vector representation using the SPECTER model.

3. **Subject Prediction**
   A multi-label classifier predicts relevant subject areas based on probability scores.

4. **Paper Recommendation**
   The system computes similarity between the query and stored paper embeddings and returns the most relevant results.

---

## Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/your-username/paperpilot.git
cd paperpilot
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the application

```
streamlit run app.py
```

---

## Deployment

The application is deployed using Streamlit Cloud.

Due to size limitations, the embedding file is not stored in the repository. Instead, it is hosted on Hugging Face and downloaded at runtime.

---

## Limitations

* The system is trained on arXiv data, so it performs best on domains such as computer science, physics, and mathematics.
* It may not perform well on domains like medicine or social sciences due to lack of training data.

---

## Future Work

* Extend support to additional domains such as biomedical research
* Improve search speed using vector indexing (e.g., FAISS)
* Enhance the user interface
* Add direct links to full research papers

---

## Author

Sanay Singh Rajawat
M.Tech (Artificial Intelligence)

---

## Summary

PaperPilot demonstrates how transformer-based embeddings and classical machine learning can be combined to build a practical and scalable research recommendation system.
