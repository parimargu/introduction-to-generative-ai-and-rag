"""
app.py — A very simple RAG (Retrieval-Augmented Generation) app.

    Upload a document  ->  chunk it  ->  embed chunks  ->  store in ChromaDB
    Ask a question     ->  embed it  ->  find similar chunks  ->  LLM answers

Run with:   streamlit run app.py
"""

import os
import uuid
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import chunking

load_dotenv()  # reads GROQ_API_KEY from a .env file if present

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # free, small, runs on CPU
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"                     # free-tier model on Groq
COLLECTION_NAME = "rag_documents"
APP_DIR = Path(__file__).parent
SAMPLE_DIR = APP_DIR / "sample_documents"
CHROMA_DIR = APP_DIR / "chroma_db"

STRATEGIES = {
    "1. Fixed-size": {
        "sample": "1_fixed_size_server_logs.txt",
        "about": "Cuts the text every N characters. Simple and predictable, "
                 "but can cut words and sentences in half.",
    },
    "2. Sentence-based": {
        "sample": "2_sentence_based_space_facts.txt",
        "about": "Groups N complete sentences into each chunk. "
                 "Never breaks a sentence in the middle.",
    },
    "3. Paragraph-based": {
        "sample": "3_paragraph_based_leave_policy.txt",
        "about": "One chunk per paragraph, following the author's own structure. "
                 "Headings stay attached to their content.",
    },
    "4. Recursive character": {
        "sample": "4_recursive_python_handbook.md",
        "about": "Tries paragraphs first, then lines, then sentences, then words, "
                 "until every chunk fits the size limit. The best general default.",
    },
    "5. Semantic": {
        "sample": "5_semantic_travel_blog.txt",
        "about": "Uses embeddings to detect where the topic changes and cuts there. "
                 "Best for long text that switches topics without clear formatting.",
    },
}

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the context "
    "provided. If the answer is not in the context, say \"I could not find that "
    "in the document.\" Keep answers short and clear, and mention which chunk "
    "numbers you used, e.g. (Chunk 2)."
)


# ---------------------------------------------------------------------------
# Cached resources (loaded once, reused on every rerun)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (first time downloads ~90 MB)...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def embed(texts):
    """Turn a list of strings into a list of vectors (normalised for cosine)."""
    model = load_embedding_model()
    return model.encode(texts, normalize_embeddings=True).tolist()


def get_collection(reset: bool = False):
    client = get_chroma_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # collection did not exist yet
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},   # cosine distance for text
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_uploaded_file(uploaded_file) -> str:
    """Extract plain text from .txt, .md or .pdf uploads."""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def chunk_text(text: str, strategy: str, params: dict):
    if strategy.startswith("1"):
        return chunking.fixed_size_chunking(text, params["chunk_size"], params["chunk_overlap"])
    if strategy.startswith("2"):
        return chunking.sentence_chunking(text, params["sentences_per_chunk"], params["overlap_sentences"])
    if strategy.startswith("3"):
        return chunking.paragraph_chunking(text, params["min_chars"], params["max_chars"])
    if strategy.startswith("4"):
        return chunking.recursive_chunking(text, params["chunk_size"], params["chunk_overlap"])
    return chunking.semantic_chunking(text, embed, params["breakpoint_percentile"], params["buffer_size"])


def add_to_vector_db(chunks, source_name: str, strategy: str, reset: bool):
    collection = get_collection(reset=reset)
    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=chunks,
        embeddings=embed(chunks),
        metadatas=[{"source": source_name, "strategy": strategy, "chunk_index": i}
                   for i in range(len(chunks))],
    )
    return collection.count()


def retrieve(question: str, top_k: int):
    collection = get_collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=embed([question]),
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, "meta": meta, "similarity": 1 - dist}
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]


def generate_answer(question: str, hits, api_key: str, model_name: str) -> str:
    if not api_key:
        return ("**No Groq API key set**, so no LLM answer was generated. "
                "The most relevant chunks are shown below — that *is* the retrieval "
                "part of RAG working. Add a key in the sidebar to get written answers.")
    from groq import Groq

    context = "\n\n".join(f"[Chunk {i + 1}]\n{h['text']}" for i, h in enumerate(hits))
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


def show_sources(hits):
    with st.expander(f"📚 Retrieved chunks ({len(hits)})"):
        for i, h in enumerate(hits, start=1):
            st.markdown(f"**Chunk {i}** · similarity `{h['similarity']:.3f}` · "
                        f"`{h['meta']['source']}` · {h['meta']['strategy']}")
            st.text(h["text"])
            st.divider()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Simple RAG · Chunking Lab", page_icon="📄", layout="wide")
st.title("📄 Simple RAG — Chunking Strategies Lab")
st.caption("Upload a document, pick a chunking strategy, then chat with your document.")

for key, default in {"messages": [], "doc_text": "", "doc_name": "", "chunks": []}.items():
    st.session_state.setdefault(key, default)

# ---------------- Sidebar: settings ----------------
with st.sidebar:
    st.header("⚙️ Chunking strategy")
    strategy = st.selectbox("Choose a strategy", list(STRATEGIES.keys()))
    st.info(STRATEGIES[strategy]["about"])

    params = {}
    if strategy.startswith(("1", "4")):
        params["chunk_size"] = st.slider("Chunk size (characters)", 100, 2000, 500, 50)
        params["chunk_overlap"] = st.slider("Chunk overlap (characters)", 0, 500, 50, 10)
    elif strategy.startswith("2"):
        params["sentences_per_chunk"] = st.slider("Sentences per chunk", 1, 20, 4)
        params["overlap_sentences"] = st.slider("Overlapping sentences", 0, 5, 1)
    elif strategy.startswith("3"):
        params["min_chars"] = st.slider("Merge paragraphs shorter than (chars)", 0, 500, 100, 10)
        params["max_chars"] = st.slider("Split paragraphs longer than (chars)", 300, 4000, 1500, 100)
    else:
        params["breakpoint_percentile"] = st.slider("Breakpoint percentile", 50, 99, 85,
                                                    help="Lower = more, smaller chunks. Higher = fewer, bigger chunks.")
        params["buffer_size"] = st.slider("Sentence buffer (neighbours)", 0, 3, 1)

    st.header("🔎 Retrieval")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, 3)

    st.header("🤖 LLM (Groq, free tier)")
    api_key = st.text_input("Groq API key", value=os.getenv("GROQ_API_KEY", ""), type="password")
    model_name = st.text_input("Model", value=os.getenv("GROQ_MODEL", DEFAULT_LLM_MODEL))

    st.divider()
    st.metric("Chunks stored in ChromaDB", get_collection().count())
    if st.button("🗑️ Clear database & chat"):
        get_collection(reset=True)
        st.session_state.update(messages=[], chunks=[])
        st.rerun()

# ---------------- Step 1: add a document ----------------
st.subheader("Step 1 · Add a document")
col_upload, col_sample = st.columns([2, 1])
with col_upload:
    uploaded = st.file_uploader("Upload a .txt, .md or .pdf file", type=["txt", "md", "pdf"])
    if uploaded is not None:
        st.session_state.doc_text = read_uploaded_file(uploaded)
        st.session_state.doc_name = uploaded.name
with col_sample:
    sample_file = STRATEGIES[strategy]["sample"]
    st.write("…or try the sample made for this strategy:")
    if st.button(f"📥 Load `{sample_file}`", width="stretch"):
        st.session_state.doc_text = (SAMPLE_DIR / sample_file).read_text(encoding="utf-8")
        st.session_state.doc_name = sample_file

if st.session_state.doc_text:
    st.success(f"Loaded **{st.session_state.doc_name}** "
               f"({len(st.session_state.doc_text):,} characters)")
    with st.expander("👀 Preview document"):
        st.text(st.session_state.doc_text[:5000])

    replace = st.checkbox("Replace documents already in the database", value=True)
    if st.button("✂️ Chunk & store in ChromaDB", type="primary"):
        try:
            with st.spinner("Chunking and embedding..."):
                chunks = chunk_text(st.session_state.doc_text, strategy, params)
                total = add_to_vector_db(chunks, st.session_state.doc_name, strategy, reset=replace)
            st.session_state.chunks = chunks
            st.success(f"Created **{len(chunks)} chunks** with *{strategy}*. "
                       f"Database now holds {total} chunks.")
        except ValueError as err:
            st.error(f"Invalid settings: {err}")

# Show the chunks so beginners can SEE what the strategy did
if st.session_state.chunks:
    chunks = st.session_state.chunks
    lengths = [len(c) for c in chunks]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks", len(chunks))
    c2.metric("Avg length", f"{sum(lengths) // len(lengths)} chars")
    c3.metric("Shortest", f"{min(lengths)} chars")
    c4.metric("Longest", f"{max(lengths)} chars")
    with st.expander("🧩 See every chunk", expanded=False):
        for i, c in enumerate(chunks, start=1):
            st.markdown(f"**Chunk {i}** — {len(c)} chars")
            st.text(c)
            st.divider()

# ---------------- Step 2: chat ----------------
st.subheader("Step 2 · Chat with your document")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

question = st.chat_input("Ask a question about the document...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        hits = retrieve(question, top_k)
        if not hits:
            answer = "The database is empty. Add a document in **Step 1** first."
        else:
            try:
                with st.spinner("Thinking..."):
                    answer = generate_answer(question, hits, api_key, model_name)
            except Exception as err:
                answer = f"⚠️ LLM call failed: `{err}`. Retrieved chunks are shown below."
        st.markdown(answer)
        if hits:
            show_sources(hits)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": hits})
