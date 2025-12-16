"""
Script to download and cache embedding model for offline use
Run this once on a machine with internet access
"""

from sentence_transformers import SentenceTransformer
import os
import shutil

# Model name
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Local cache directory
CACHE_DIR = "./models/embeddings"

def download_model():
    """
    Download the embedding model and save to local directory
    """
    print("=" * 60)
    print("📥 Downloading Embedding Model")
    print("=" * 60)
    
    try:
        print(f"\n1. Downloading {MODEL_NAME}...")
        print("   This may take a few minutes...")
        
        # Download model (will cache automatically)
        model = SentenceTransformer(MODEL_NAME)
        
        print("   ✓ Model downloaded successfully")
        
        # Get cache location
        cache_folder = model._model_card_vars.get('cache_folder', 
                       os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'sentence_transformers'))
        
        print(f"\n2. Model cached at:")
        print(f"   {cache_folder}")
        
        # Create local copy
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Copy to project directory
        model_path = os.path.join(CACHE_DIR, "all-MiniLM-L6-v2")
        
        if os.path.exists(model_path):
            print(f"\n3. Removing old local copy...")
            shutil.rmtree(model_path)
        
        print(f"\n3. Creating local copy at:")
        print(f"   {model_path}")
        
        # Save model to local directory
        model.save(model_path)
        
        print("\n✅ SUCCESS! Model ready for offline use")
        print("\n📋 Next steps:")
        print("   1. Copy the entire 'models/embeddings' folder to your offline machine")
        print("   2. Use the offline RAG configuration (see rag_offline.py)")
        
        # Verify model works
        print("\n🧪 Testing model...")
        test_embedding = model.encode(["test sentence"])
        print(f"   ✓ Model works! Embedding shape: {test_embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("   - Check internet connection")
        print("   - Try: pip install --upgrade sentence-transformers")
        print("   - Try: pip install --upgrade transformers")
        return False


def verify_offline_model():
    """
    Verify that offline model works
    """
    model_path = os.path.join(CACHE_DIR, "all-MiniLM-L6-v2")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return False
    
    try:
        print(f"\n✓ Loading offline model from {model_path}...")
        model = SentenceTransformer(model_path)
        
        test_embedding = model.encode(["test"])
        print(f"✓ Model loaded successfully! Shape: {test_embedding.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        print("🔍 Verifying offline model...")
        verify_offline_model()
    else:
        print("🌐 Requires internet connection!")
        print("   Run this script on a machine with internet access")
        print("   Then copy the 'models/embeddings' folder to your offline machine\n")
        
        response = input("Continue? (y/n): ")
        if response.lower() == 'y':
            success = download_model()
            
            if success:
                print("\n" + "=" * 60)
                print("🎉 Setup Complete!")
                print("=" * 60)
                print("\nTo use offline:")
                print("   1. Copy 'models/embeddings' to offline machine")
                print("   2. Update rag.py to use local path")
                print("   3. Run: python offline_model_setup.py --verify")