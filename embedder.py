from sentence_transformers import SentenceTransformer
import os
import json

# runs on your laptop - completely free, no API key needed
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(all_chunks):
    print(f"Embedding {len(all_chunks)} chunks...")
    texts = [chunk['text'] for chunk in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    embedded_data = []
    for i, chunk in enumerate(all_chunks):
        embedded_data.append({
            "text": chunk['text'],
            "embedding": embeddings[i].tolist(),
            "source": chunk['source'],
            "chunk_id": chunk['chunk_id']
        })
    
    print(f"Successfully embedded {len(embedded_data)} chunks!")
    return embedded_data

def save_embeddings(embedded_data, output_file="embeddings.json"):
    with open(output_file, "w") as f:
        json.dump(embedded_data, f)
    print(f"Saved {len(embedded_data)} embeddings to {output_file}")

if __name__ == "__main__":
    from pdf_reader import read_all_pdfs
    from chunker import chunk_all_documents
    
    print("=== STEP 1: Reading PDFs ===")
    documents = read_all_pdfs("data")
    
    print("\n=== STEP 2: Chunking ===")
    chunks = chunk_all_documents(documents)
    print(f"Total chunks: {len(chunks)}")
    
    print("\n=== STEP 3: Embedding (local - free) ===")
    embedded_data = embed_chunks(chunks)
    
    print("\n=== STEP 4: Saving ===")
    save_embeddings(embedded_data)
    
    print("\n=== DONE ===")
    print(f"First embedding preview (first 5 numbers):")
    print(embedded_data[0]['embedding'][:5])
    print(f"Each chunk = {len(embedded_data[0]['embedding'])} numbers")