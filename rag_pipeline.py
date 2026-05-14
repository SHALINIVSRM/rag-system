from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from dotenv import load_dotenv

load_dotenv()

# models
embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# model config - change only this line to switch
GROQ_MODEL = "llama-3.3-70b-versatile"

def load_embeddings(file="embeddings.json"):
    with open(file, "r") as f:
        data = json.load(f)
    return data

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def search(question, all_data, top_k=3):
    question_embedding = embedding_model.encode([question])[0]
    all_embeddings = np.array([item['embedding'] for item in all_data])
    question_vec = np.array(question_embedding)
    norms = np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(question_vec)
    scores = np.dot(all_embeddings, question_vec) / norms
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(scores[i], all_data[i]) for i in top_indices]

def build_context(results):
    context = ""
    for i, (score, item) in enumerate(results):
        context += f"[Source: {item['source']}]\n{item['text']}\n\n"
    return context

def ask(question, all_data):
    # step 1: find relevant chunks
    results = search(question, all_data)
    
    # step 2: build context from chunks
    context = build_context(results)
    
    # step 3: build prompt
    prompt = f"""You are a helpful study assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I could not find this in the provided documents."
Always mention which source document you used.

Context:
{context}

Question: {question}

Answer:"""
    
    # step 4: ask Groq
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    
    # step 5: return answer with sources
    sources = list(set([item['source'] for _, item in results]))
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "top_score": float(results[0][0])
    }

if __name__ == "__main__":
    print("Loading embeddings...")
    all_data = load_embeddings()
    print(f"Loaded {len(all_data)} embeddings\n")
    
    # test questions
    questions = [
        "what is a primary key?",
        "explain ACID properties of transactions",
        "what is normalization and why is it important?"
    ]
    
    for question in questions:
        print(f"Q: {question}")
        print("-" * 50)
        result = ask(question, all_data)
        print(f"A: {result['answer']}")
        print(f"\nSources: {result['sources']}")
        print(f"Confidence: {result['top_score']:.3f}")
        print("=" * 60 + "\n")