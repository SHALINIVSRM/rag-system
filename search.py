import numpy as np
import json
from sentence_transformers import SentenceTransformer

# same model as embedder - must match!
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def load_embeddings(file="embeddings.json"):
    with open(file, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} embeddings from disk")
    return data

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def search(question, all_data, top_k=3):
    print(f"\nSearching for: '{question}'")
    question_embedding = model.encode([question])[0]
    
    # vectorized - compare all at once instead of one by one
    all_embeddings = np.array([item['embedding'] for item in all_data])
    question_vec = np.array(question_embedding)
    
    # cosine similarity for all chunks at once
    norms = np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(question_vec)
    scores = np.dot(all_embeddings, question_vec) / norms
    
    # get top k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    return [(scores[i], all_data[i]) for i in top_indices]
def print_results(results):
    print("\n=== TOP RESULTS ===")
    for i, (score, item) in enumerate(results):
        print(f"\nResult {i+1} — Score: {score:.3f}")
        print(f"Source: {item['source']}")
        print(f"Text preview: {item['text'][:200]}...")
        print("-" * 50)

if __name__ == "__main__":
    # load saved embeddings
    all_data = load_embeddings()
    
    # test with real DBMS questions
    questions = [
        "what is a primary key?",
        "explain transaction management",
        "what is normalization in database?"
    ]
    
    for question in questions:
        results = search(question, all_data)
        print_results(results)
        print("\n" + "="*60 + "\n")