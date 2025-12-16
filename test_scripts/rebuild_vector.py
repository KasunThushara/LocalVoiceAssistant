"""
Offline Vector Store Builder
Uses local embedding model - no internet required
"""

import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # Changed!
from langchain_community.document_loaders import TextLoader

DOC_PATH = "docs/seeed.txt"
VECTOR_PATH = "vector_store/faiss_index"
LOCAL_MODEL_PATH = "models/embeddings/all-MiniLM-L6-v2"  # Local path

def rebuild_vector_store():
    """
    Force rebuild of vector store with offline model
    """
    print("=" * 80)
    print("🔧 REBUILDING VECTOR STORE (OFFLINE MODE)")
    print("=" * 80)
    
    # Step 1: Check if local model exists
    print(f"\n🔍 Checking for local embedding model...")
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"   ❌ ERROR: Local model not found at {LOCAL_MODEL_PATH}")
        print("\n💡 Options:")
        print("   1. Download model with: python offline_model_setup.py")
        print("   2. Or copy from another machine with internet")
        print("   3. Or use online mode (requires internet)")
        return False
    else:
        print(f"   ✅ Found local model at {LOCAL_MODEL_PATH}")
    
    # Step 2: Delete old vector store
    if os.path.exists("vector_store"):
        print("\n🗑️  Deleting old vector store...")
        shutil.rmtree("vector_store")
        print("   ✅ Deleted")
    
    # Step 3: Load document
    print("\n📄 Loading document...")
    if not os.path.exists(DOC_PATH):
        print(f"   ❌ ERROR: Document not found at {DOC_PATH}")
        return False
    
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    documents = loader.load()
    print(f"   ✅ Loaded {len(documents)} documents")
    print(f"   📊 Total characters: {sum(len(doc.page_content) for doc in documents)}")
    
    # Step 4: Split into chunks
    print("\n✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=650,  # Optimized for Q&A pairs
        chunk_overlap=80,
        separators=["\nQ:", "\n\n", "\n", ". ", " ", ""]
    )
    texts = splitter.split_documents(documents)
    print(f"   ✅ Created {len(texts)} chunks")
    
    # Show sample chunks
    print("\n📋 Sample chunks:")
    for i, text in enumerate(texts[:3]):
        print(f"\n   Chunk {i+1} (length: {len(text.page_content)}):")
        print(f"   {text.page_content[:150]}...")
    
    # Step 5: Load embeddings from LOCAL model
    print("\n🧠 Loading embedding model from local path...")
    print(f"   Path: {LOCAL_MODEL_PATH}")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_MODEL_PATH,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("   ✅ Model loaded successfully (OFFLINE)")
    except Exception as e:
        print(f"   ❌ ERROR loading model: {e}")
        print("\n💡 Try:")
        print("   1. Run: python offline_model_setup.py")
        print("   2. Check model files exist in models/embeddings/")
        return False
    
    # Step 6: Build FAISS index
    print("\n🏗️  Building FAISS index...")
    print("   ⏳ This may take 30-60 seconds...")
    
    try:
        vectorstore = FAISS.from_documents(texts, embeddings)
        print("   ✅ Index built successfully")
    except Exception as e:
        print(f"   ❌ ERROR building index: {e}")
        return False
    
    # Step 7: Save
    print("\n💾 Saving vector store...")
    os.makedirs(os.path.dirname(VECTOR_PATH), exist_ok=True)
    vectorstore.save_local(VECTOR_PATH)
    print(f"   ✅ Saved to {VECTOR_PATH}")
    
    # Calculate size
    total_size = sum(
        os.path.getsize(os.path.join(VECTOR_PATH, f)) 
        for f in os.listdir(VECTOR_PATH)
    )
    print(f"   📦 Vector store size: {total_size / 1024:.1f} KB")
    
    # Step 8: Test with sample queries
    print("\n🧪 Testing with sample queries...")
    
    test_queries = [
        "What is Seeed Studio Vision Solutions?",
        "Tell me about reSpeaker",
        "What products does Seeed offer?"
    ]
    
    for query in test_queries:
        results = vectorstore.similarity_search_with_score(query, k=3)
        
        print(f"\n   Query: '{query}'")
        if results:
            best_score = results[0][1]
            print(f"   Best match score: {best_score:.4f}")
            
            if best_score < 1.5:
                print(f"   ✅ Good match! (< 1.5)")
            elif best_score < 2.5:
                print(f"   ⚠️  Acceptable (< 2.5)")
            else:
                print(f"   ❌ Poor match (> 2.5)")
    
    print("\n" + "=" * 80)
    print("✅ REBUILD COMPLETE!")
    print("=" * 80)
    
    print("\n📊 Summary:")
    print(f"   • Documents loaded: {len(documents)}")
    print(f"   • Chunks created: {len(texts)}")
    print(f"   • Model: {LOCAL_MODEL_PATH}")
    print(f"   • Vector store: {VECTOR_PATH}")
    print(f"   • Mode: OFFLINE ✓")
    
    return True


def verify_setup():
    """
    Verify that everything is set up correctly
    """
    print("\n🔍 Verifying setup...")
    
    issues = []
    
    # Check model
    if not os.path.exists(LOCAL_MODEL_PATH):
        issues.append(f"❌ Local model not found: {LOCAL_MODEL_PATH}")
    else:
        print(f"✅ Model exists: {LOCAL_MODEL_PATH}")
    
    # Check docs
    if not os.path.exists(DOC_PATH):
        issues.append(f"❌ Document not found: {DOC_PATH}")
    else:
        print(f"✅ Document exists: {DOC_PATH}")
    
    # Check vector store
    if os.path.exists(VECTOR_PATH):
        print(f"✅ Vector store exists: {VECTOR_PATH}")
    else:
        print(f"⚠️  Vector store not built yet: {VECTOR_PATH}")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"   {issue}")
        return False
    
    print("\n✅ All checks passed!")
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_setup()
    else:
        print("🌐 OFFLINE Vector Store Builder")
        print("   No internet connection required!\n")
        
        # Verify setup first
        if not verify_setup():
            print("\n❌ Setup incomplete. Fix issues above and try again.")
            sys.exit(1)
        
        print("\n" + "=" * 80)
        response = input("Ready to rebuild vector store? (y/n): ")
        
        if response.lower() == 'y':
            success = rebuild_vector_store()
            
            if success:
                print("\n💡 Next steps:")
                print("   1. Make sure rag.py uses HuggingFaceEmbeddings")
                print("   2. Restart your Flask app: python app.py")
                print("   3. Test queries to verify it works")
            else:
                print("\n❌ Rebuild failed. Check error messages above.")
                sys.exit(1)
        else:
            print("\n👋 Cancelled. Run again when ready.")