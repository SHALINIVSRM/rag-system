import pdfplumber
import PyPDF2
import os

def read_pdf(file_path):
    text = ""
    
    # Try pdfplumber first (handles more PDF types)
    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")
            for page_num, page in enumerate(pdf.pages):
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
                    print(f"Page {page_num + 1} read ✓")
                else:
                    print(f"Page {page_num + 1} - no text (image page)")
    except Exception as e:
        print(f"pdfplumber failed: {e}, trying PyPDF2...")
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    
    return text

def read_all_pdfs(folder_path):
    all_documents = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDFs in {folder_path}")
    
    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        print(f"\nReading: {pdf_file}")
        text = read_pdf(full_path)
        all_documents.append({
            "filename": pdf_file,
            "text": text,
            "total_chars": len(text)
        })
        print(f"Extracted {len(text)} characters from {pdf_file}")
    
    return all_documents

if __name__ == "__main__":
    documents = read_all_pdfs("data")
    print(f"\n=== SUMMARY ===")
    for doc in documents:
        if doc['total_chars'] == 0:
            print(f"❌ {doc['filename']}: 0 characters - scanned image PDF, cannot read")
        else:
            print(f"✅ {doc['filename']}: {doc['total_chars']} characters")