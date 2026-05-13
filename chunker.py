def chunk_text(text, chunk_size=150, overlap=20):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def chunk_all_documents(documents):
    all_chunks = []
    
    for doc in documents:
        if doc['total_chars'] == 0:
            continue
            
        chunks = chunk_text(doc['text'])
        print(f"{doc['filename']}: {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc['filename'],
                "chunk_id": i
            })
    
    return all_chunks

if __name__ == "__main__":
    from pdf_reader import read_all_pdfs
    
    documents = read_all_pdfs("data")
    print("\n=== CHUNKING ===")
    chunks = chunk_all_documents(documents)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total chunks: {len(chunks)}")
    print(f"\nFirst chunk preview:")
    print(chunks[0]['text'][:200])
    print(f"\nSource: {chunks[0]['source']}")