import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# -------------------------
# LOAD ENV VARIABLES
# -------------------------
load_dotenv()

# SAFE API KEY LOADING (NO ERROR)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

try:
    if not GROQ_API_KEY:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    pass

if not GROQ_API_KEY:
    st.error("❌ GROQ API KEY missing. Add in .env or Streamlit secrets.")
    st.stop()

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("💬 Chat with your PDF")

# -------------------------
# CACHE EMBEDDINGS
# -------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# -------------------------
# PROCESS PDF (FIXED VERSION 🔥)
# -------------------------
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Check if PDF has text
    if not docs:
        st.error("❌ No text found in PDF")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        st.error("❌ Could not split PDF into chunks")
        st.stop()

    # Clean text
    texts = [doc.page_content.strip() for doc in chunks if doc.page_content.strip()]

    if not texts:
        st.error("❌ PDF has no readable text (maybe scanned image PDF)")
        st.stop()

    # Create vector DB safely
    vectordb = Chroma.from_texts(texts, embeddings)
    return vectordb

# -------------------------
# LLM
# -------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY
)

# -------------------------
# SESSION STATE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

    st.markdown("---")
    st.markdown("### ℹ️ Instructions")
    st.markdown("1. Upload PDF\n2. Ask questions\n3. Chat continues")

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing PDF..."):
        st.session_state.vectordb = process_pdf("temp.pdf")

    st.success("✅ PDF Ready!")

# -------------------------
# CHAT DISPLAY
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -------------------------
# USER INPUT
# -------------------------
if prompt := st.chat_input("Ask something about your PDF..."):

    if st.session_state.vectordb is None:
        st.warning("⚠️ Upload a PDF first")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    # -------------------------
    # RETRIEVE CONTEXT
    # -------------------------
    docs = st.session_state.vectordb.similarity_search(prompt, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # -------------------------
    # FINAL PROMPT
    # -------------------------
    messages = [
        SystemMessage(content="Answer ONLY from the provided PDF context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{prompt}")
    ]

    # -------------------------
    # RESPONSE
    # -------------------------
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):
            response = llm.invoke(messages)
            answer = response.content
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})