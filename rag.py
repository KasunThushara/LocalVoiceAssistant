import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

DOC_PATH = "docs/seeed.txt"
VECTOR_PATH = "vector_store/faiss_index"
LOCAL_MODEL_PATH = "models/embeddings/all-MiniLM-L6-v2"

# Global cache for embeddings and vector store
_embeddings = None
_vectorstore = None

def get_embeddings():
    """Singleton pattern for embeddings model."""
    global _embeddings
    if _embeddings is None:
        print(f"[RAG] Loading embeddings from local path: {LOCAL_MODEL_PATH}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_MODEL_PATH,  # Local path instead of downloading
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("[RAG] ✅ Embeddings loaded from local model (no internet needed)")
    return _embeddings

def load_vector_db():
    """Load or create vector database with caching."""
    global _vectorstore
    
    if _vectorstore is not None:
        return _vectorstore
    
    embeddings = get_embeddings()
    
    if os.path.exists(VECTOR_PATH):
        print("[RAG] Loading existing FAISS index...")
        _vectorstore = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
        return _vectorstore
    
    print("[RAG] Creating new FAISS index...")
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    documents = loader.load()
    
    # Optimized chunking for Q&A format
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Increased to capture full Q&A pairs
        chunk_overlap=100,  # More overlap for better matching
        separators=["\n\n", "\nQ:", "\n", ". ", " ", ""]
    )
    texts = splitter.split_documents(documents)
    
    _vectorstore = FAISS.from_documents(texts, embeddings)
    os.makedirs(os.path.dirname(VECTOR_PATH), exist_ok=True)
    _vectorstore.save_local(VECTOR_PATH)
    
    print(f"[RAG] Created index with {len(texts)} chunks")
    return _vectorstore

def is_product_query(query: str) -> bool:
    """
    Check if query is about Seeed products using keyword matching.
    This helps catch queries that should use RAG even with lower similarity.
    """
    query_lower = query.lower()
    
    # Product/brand keywords
    product_keywords = [
        "seeed", "respeaker", "sensecraft", "fusion",
        "vision solution", "voice solution", "sensor network", "robotics solution",
        "microphone", "mic array", "camera", "ai camera",
        "jetson", "raspberry pi", "edge computer",
        "robot arm", "lidar", "imu", "servo"
    ]
    
    # Check if any keyword is in query
    return any(keyword in query_lower for keyword in product_keywords)

def retrieve_relevant(query: str, k: int = 4):
    """
    Retrieve relevant context with improved matching logic.
    Returns context and relevance flag.
    
    FAISS L2 distance scores:
    - 0.0-0.8: Excellent match
    - 0.8-1.5: Good match
    - 1.5-2.5: Acceptable match
    - 2.5+: Poor match
    """
    db = load_vector_db()
    
    # Normalize query for better matching
    query_normalized = query.strip()
    
    # Get documents with scores
    docs_with_scores = db.similarity_search_with_score(query_normalized, k=k)
    
    # Debug logging
    if docs_with_scores:
        best_score = docs_with_scores[0][1]
        print(f"[RAG] Query: '{query_normalized}'")
        print(f"[RAG] Best match score: {best_score:.4f}")
        print(f"[RAG] Top match: {docs_with_scores[0][0].page_content[:100]}...")
    
    # Multi-factor relevance check
    is_product_related = is_product_query(query_normalized)
    
    # Threshold based on FAISS L2 distance
    # Lower scores = better matches
    has_excellent_match = len(docs_with_scores) > 0 and docs_with_scores[0][1] < 1.2
    has_good_match = len(docs_with_scores) > 0 and docs_with_scores[0][1] < 2.0
    
    # Decision logic:
    # 1. Always use RAG for excellent matches
    # 2. Use RAG for good matches if product-related
    # 3. Use RAG if product keywords detected (even with medium scores)
    if has_excellent_match:
        is_relevant = True
        reason = f"excellent match (score={docs_with_scores[0][1]:.4f})"
    elif has_good_match and is_product_related:
        is_relevant = True
        reason = f"good match + product keywords (score={docs_with_scores[0][1]:.4f})"
    elif is_product_related:
        is_relevant = True
        reason = f"product keywords detected (score={docs_with_scores[0][1]:.4f})"
    else:
        is_relevant = False
        reason = f"no match (score={docs_with_scores[0][1]:.4f})"
    
    if is_relevant:
        # Take top results
        context = "\n\n".join([doc.page_content for doc, score in docs_with_scores[:3]])
        print(f"[RAG] âœ… Using RAG: {reason}")
        return context, True
    
    print(f"[RAG] âŒ Using general response: {reason}")
    return "", False

def build_rag_prompt(query: str, context: str) -> str:
    """Build optimized prompt for concise, specific Seeed Studio responses."""
    return f"""You are a Seeed Studio product expert. Answer the customer's question using the information provided below.

Product Information:
{context}

Customer Question: {query}

INSTRUCTIONS:
1. The information above DOES answer the question - use it directly
2. Answer in 2-3 clear sentences maximum
3. Be specific - mention product names, numbers, and key features
4. Use friendly, conversational tone
5. Do NOT say "I don't have information" - the context above has what you need
6. Paraphrase the information naturally, don't just copy word-for-word

Examples:
Q: What are vision solutions?
Good: "Seeed provides AI cameras ranging from 1-4 TOPS for efficient tasks up to 275 TOPS edge computers for demanding scenarios. Our SenseCraft platform offers 300+ pre-trained models for easy deployment across retail, security, agriculture, and robotics."

Q: What is reSpeaker?
Good: "reSpeaker is our voice solutions product line with 2-mic and 4-mic arrays for high-quality audio capture. The arrays include AI algorithms like AEC, beamforming, and echo cancellation for clear voice interaction."

Now answer the customer's question in 2-3 sentences:"""

def build_general_prompt(query: str) -> str:
    """Build prompt for non-product general questions."""
    return f"""Answer this question briefly and helpfully in 2-3 sentences.

Question: {query}

Be conversational, friendly, and concise.

Answer:"""

# Initialize on import (optional - for faster first query)
def init_rag():
    """Pre-load vector store at startup."""
    print("[RAG] Initializing vector database...")
    load_vector_db()
    print("[RAG] Ready!")