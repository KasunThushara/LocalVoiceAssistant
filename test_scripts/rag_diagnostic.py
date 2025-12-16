"""
RAG Diagnostic Tool - Test why queries aren't matching
Run this to debug your RAG retrieval issues
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import TextLoader

DOC_PATH = "docs/seeed.txt"
VECTOR_PATH = "vector_store/faiss_index"

def diagnose_query(query: str):
    """
    Diagnose why a query might not be matching well
    """
    print("=" * 80)
    print(f"DIAGNOSING QUERY: '{query}'")
    print("=" * 80)
    
    # Load embeddings and vector store
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists(VECTOR_PATH):
        print("❌ Vector store not found. Creating new one...")
        loader = TextLoader(DOC_PATH, encoding="utf-8")
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        )
        texts = splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(texts, embeddings)
        os.makedirs(os.path.dirname(VECTOR_PATH), exist_ok=True)
        vectorstore.save_local(VECTOR_PATH)
    else:
        vectorstore = FAISS.load_local(VECTOR_PATH, embeddings, 
                                       allow_dangerous_deserialization=True)
    
    # Get top results with scores
    print("\n📊 TOP 5 RESULTS WITH SIMILARITY SCORES:")
    print("-" * 80)
    
    results = vectorstore.similarity_search_with_score(query, k=5)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n#{i} - Score: {score:.4f} {'✅ PASS' if score < 0.6 else '❌ FAIL'} (threshold: 0.6)")
        print(f"Content preview: {doc.page_content[:200]}...")
        print("-" * 80)
    
    # Test different thresholds
    print("\n🎯 THRESHOLD ANALYSIS:")
    print("-" * 80)
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
    for thresh in thresholds:
        passing = sum(1 for _, score in results if score < thresh)
        print(f"Threshold {thresh}: {passing}/5 results would pass")
    
    # Recommend threshold
    best_score = results[0][1] if results else 1.0
    print(f"\n💡 RECOMMENDATION:")
    print(f"   Best match score: {best_score:.4f}")
    
    if best_score < 0.5:
        print(f"   ✅ Good match! Current threshold (0.6) should work.")
    elif best_score < 0.7:
        print(f"   ⚠️  Weak match. Consider raising threshold to 0.7 or 0.75")
    else:
        print(f"   ❌ Poor match. Query needs better correction or docs need improvement.")
    
    print("\n" + "=" * 80)
    return results


if __name__ == "__main__":
    # Test cases
    test_queries = [
        "What is Seeed Studio vision Solutions?",
        "What is Seeed Studio Vision Solutions?",
        "Tell me about vision solutions",
        "What are the voice solutions?",
        "What is reSpeaker?",
    ]
    
    for query in test_queries:
        diagnose_query(query)
        print("\n\n")