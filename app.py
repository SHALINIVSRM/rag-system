import streamlit as st
from rag_pipeline import ask, load_embeddings, clear_history
from sentence_transformers import SentenceTransformer

# page config
st.set_page_config(
    page_title="DBMS Study Assistant",
    page_icon="📚",
    layout="centered"
)

# password protection
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔒 DBMS Study Assistant")
        pwd = st.text_input("Enter password to access", type="password")
        if st.button("Login"):
            if pwd == "dbms2025":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Wrong password")
        st.stop()

check_password()

# load embeddings once
@st.cache_resource(show_spinner="Loading your DBMS notes...")
def load_data():
    return load_embeddings()

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    return SentenceTransformer('BAAI/bge-small-en-v1.5')

# sidebar
with st.sidebar:
    st.title("📚 DBMS Assistant")
    st.markdown("---")
    st.markdown("**Documents loaded:**")
    st.markdown("- DBMS UNIT 1.pdf")
    st.markdown("- DBMS UNIT 2,3.pdf")
    st.markdown("- UNIT-IV TRANSACTION MANAGEMENT.pdf")
    st.markdown("---")
    st.markdown("**Total chunks:** 357")
    st.markdown("**Model:** Llama 3.3 70B")
    st.markdown("**Embeddings:** BGE Small")
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        clear_history()
        st.rerun()
    st.markdown("---")
    st.markdown("**Sample questions:**")
    st.markdown("- What is a primary key?")
    st.markdown("- Explain ACID properties")
    st.markdown("- What is normalization?")
    st.markdown("- What is a foreign key?")

# main title
st.title("📚 DBMS Study Assistant")
st.caption("Ask questions about your DBMS notes — answers come directly from your documents")

# load data
all_data = load_data()

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I'm your DBMS study assistant. I've read your notes and I'm ready to answer questions. What would you like to know? 📖"
    })

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "source" in msg:
            st.caption(f"📄 Source: {msg['source']} | Confidence: {msg['confidence']:.2f}")

# chat input
if question := st.chat_input("Ask a question about DBMS..."):
    # show user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.write(question)
    
    # get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching your notes..."):
            result = ask(question, all_data)
        
        st.write(result['answer'])
        st.caption(f"📄 Source: {', '.join(result['sources'])} | Confidence: {result['top_score']:.2f}")
    
    # save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result['answer'],
        "source": ', '.join(result['sources']),
        "confidence": result['top_score']
    })