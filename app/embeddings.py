import os
import numpy as np
import faiss
import openai
import pickle

EMBED_DIM = 1536

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, 'vectorstore_faiss.faiss')
EMBEDDING_STORE_PATH = os.path.join(BASE_DIR, 'embedding_store.pkl')

embedding_store = {}

def save_embedding_store():
    with open(EMBEDDING_STORE_PATH, 'wb') as f:
        pickle.dump(embedding_store, f)

def load_embedding_store():
    global embedding_store
    if os.path.exists(EMBEDDING_STORE_PATH):
        with open(EMBEDDING_STORE_PATH, 'rb') as f:
            embedding_store = pickle.load(f)
    else:
        embedding_store = {}

def create_faiss_index():
    index_flat = faiss.IndexFlatL2(EMBED_DIM)
    return faiss.IndexIDMap(index_flat)

def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index.")
        return faiss.read_index(FAISS_INDEX_PATH)
    else:
        print("Creating new FAISS index.")
        return create_faiss_index()

def save_faiss_index(index):
    faiss.write_index(index, FAISS_INDEX_PATH)

def get_embedding(text: str) -> np.ndarray:
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32).reshape(1, -1)

def add_embedding(index, embedding: np.ndarray, embedding_id: int):
    index.add_with_ids(embedding, np.array([embedding_id], dtype=np.int64))
    embedding_store[embedding_id] = embedding.flatten()
    save_faiss_index(index)
    save_embedding_store()

def get_resume_embedding(embedding_id: int) -> np.ndarray:
    return embedding_store.get(embedding_id)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def search_embeddings(index, query_embedding: np.ndarray, top_k: int = 5):
    D, I = index.search(query_embedding, top_k)
    return I[0], D[0]

faiss_index = load_faiss_index()
load_embedding_store()
