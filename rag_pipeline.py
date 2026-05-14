from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from dotenv import load_dotenv

load_dotenv()

embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"

# conversation memory
chat_history = []

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
    for score, item in results:
        context += f"[Source: {item['source']}]\n{item['text']}\n\n"
    return context

def ask(question, all_data):
    global chat_history
    
    # step 1: find relevant chunks
    results = search(question, all_data)
    context = build_context(results)
    
    # step 2: build messages with history
    messages = []
    
    # system message
    messages.append({
        "role": "system",
        "content": """You are a helpful study assistant for DBMS. 
Answer questions using ONLY the context provided.
If the answer is not in the context say 'I could not find this in the provided documents.'
Always mention which source document you used.
Keep answers clear and concise."""
    })
    
    # add last 3 conversations as memory
    for prev_q, prev_a in chat_history[-3:]:
        messages.append({"role": "user", "content": prev_q})
        messages.append({"role": "assistant", "content": prev_a})
    
    # add current question with context
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    })
    
    # step 3: ask Groq
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    
    # step 4: save to memory
    chat_history.append((question, answer))
    
    sources = list(set([item['source'] for _, item in results]))
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "top_score": float(results[0][0])
    }

def clear_history():
    global chat_history
    chat_history = []
    print("Chat history cleared!")

if __name__ == "__main__":
    print("Loading embeddings...")
    all_data = load_embeddings()
    print(f"Loaded {len(all_data)} embeddings")
    print("\n🤖 DBMS Study Assistant Ready!")
    print("Type 'quit' to exit | 'clear' to reset memory\n")
    
    while True:
        question = input("You: ").strip()
        
        if not question:
            continue
        if question.lower() == 'quit':
            print("Goodbye!")
            break
        if question.lower() == 'clear':
            clear_history()
            continue
        
        result = ask(question, all_data)
        print(f"\nAssistant: {result['answer']}")
        print(f"Source: {result['sources']}")
        print(f"Confidence: {result['top_score']:.3f}\n")