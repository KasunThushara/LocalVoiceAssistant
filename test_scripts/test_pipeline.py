"""
Complete Pipeline Test - Test STT correction + RAG retrieval + prompt quality
Run this before deploying to catch issues early
"""

from stt_corrector import correct_stt_text
from rag import retrieve_relevant, build_rag_prompt, build_general_prompt, init_rag

def test_full_pipeline(stt_input: str):
    """
    Test the complete pipeline for a given STT input
    """
    print("\n" + "=" * 90)
    print(f"🎤 ORIGINAL STT INPUT: '{stt_input}'")
    print("=" * 90)
    
    # Step 1: Correction
    corrected = correct_stt_text(stt_input, use_llm=False)
    if corrected != stt_input:
        print(f"✅ CORRECTED TO: '{corrected}'")
    else:
        print(f"ℹ️  NO CORRECTION NEEDED")
    
    # Step 2: RAG Retrieval
    context, is_relevant = retrieve_relevant(corrected, k=4)
    
    print(f"\n📚 RAG RESULT: {'✅ RELEVANT' if is_relevant else '❌ NOT RELEVANT'}")
    
    if is_relevant:
        print(f"\n📄 RETRIEVED CONTEXT (preview):")
        print("-" * 90)
        print(context[:500] + "..." if len(context) > 500 else context)
        print("-" * 90)
        
        # Step 3: Build prompt
        prompt = build_rag_prompt(corrected, context)
        print(f"\n💬 PROMPT TO LLM:")
        print("-" * 90)
        print(prompt)
        print("-" * 90)
    else:
        prompt = build_general_prompt(corrected)
        print(f"\n💬 GENERAL PROMPT:")
        print("-" * 90)
        print(prompt)
        print("-" * 90)
    
    print("\n" + "=" * 90)


def run_comprehensive_tests():
    """
    Run a comprehensive test suite
    """
    print("\n" + "🧪" * 40)
    print("COMPREHENSIVE PIPELINE TESTING")
    print("🧪" * 40)
    
    test_cases = [
        # Product queries with typos
        ("What is seat studio vision solutions?", "Should correct 'seat' and match vision docs"),
        ("Tell me about re speaker", "Should correct to reSpeaker"),
        ("What are seed voice solutions?", "Should correct 'seed' and match voice docs"),
        ("Does seeds studio support raspberry pie?", "Should correct both 'seeds' and 'pie'"),
        
        # Correct queries
        ("What is Seeed Studio Vision Solutions?", "Should match directly"),
        ("What is reSpeaker?", "Should match directly"),
        ("Tell me about SenseCraft", "Should match directly"),
        
        # Technical queries
        ("What 4 mike arrays does seed offer?", "Should correct '4 mike' and 'seed'"),
        ("What are the specs of respeaker 4 mic array?", "Should find technical specs"),
        
        # General queries (should NOT use RAG)
        ("What is the weather today?", "Should use general response"),
        ("How are you?", "Should use general response"),
    ]
    
    for stt_input, description in test_cases:
        print(f"\n📝 TEST: {description}")
        test_full_pipeline(stt_input)
        input("\nPress Enter to continue to next test...")


if __name__ == "__main__":
    # Initialize RAG first
    print("Initializing RAG system...")
    init_rag()
    print("✅ RAG initialized\n")
    
    # Run tests
    choice = input("Run [C]omprehensive tests or [S]ingle query? (C/S): ").strip().upper()
    
    if choice == 'C':
        run_comprehensive_tests()
    else:
        query = input("\nEnter your test query: ").strip()
        test_full_pipeline(query)
    
    print("\n✅ Testing complete!")
