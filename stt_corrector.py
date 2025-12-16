import re
from difflib import SequenceMatcher

# ============================================
# KEYWORD CORRECTION DICTIONARIES (COMPREHENSIVE)
# ============================================

BRAND_CORRECTIONS = {
    # Seeed Studio variations (most common)
    "seed studio": "Seeed Studio",
    "seed studios": "Seeed Studio",
    "seat studio": "Seeed Studio",
    "seat studios": "Seeed Studio",
    "sid studio": "Seeed Studio",
    "c studio": "Seeed Studio",
    "cd studio": "Seeed Studio",
    "c-studio": "Seeed Studio",
    "c to do your": "Seeed Studio",
    "c to do": "Seeed Studio",
    "c2d-o": "Seeed Studio",
    "c2do": "Seeed Studio",
    "c 2 d o": "Seeed Studio",
    "seeds studio": "Seeed Studio",
    "see studio": "Seeed Studio",
    "see studios": "Seeed Studio",
    "sea studio": "Seeed Studio",
    "cs studio": "Seeed Studio",
    "sits studio": "Seeed Studio",
    "is your studio":"Seeed Studio",
    "sit studio": "Seeed Studio",
    "his studio": "Seeed Studio",
    "is studio": "Seeed Studio",
    "state studio": "Seeed Studio",
    "the studio": "Seeed Studio",
    "ic studio": "Seeed Studio",
    "these studio": "Seeed Studio",
    "this studio": "Seeed Studio",
    "cease studio": "Seeed Studio",
    "its studio":"Seeed Studio",
    "seeds": "Seeed",
    "seed": "Seeed",
    "seat": "Seeed",
    "sid": "Seeed",
    "cids": "Seeed",
    "cease": "Seeed",
    "it studio":"Seeed Studio", 
    "seats to your":"Seeed Studio",
    "C's" : "Seeed",
    "C-T-D-O":"Seeed Studio",
    "C-T-S to D-O":"Seeed Studio",
    "it study":"Seeed Studio",
    "C-T2DO":"Seeed Studio",
    "but is:":"What is",
    "Seeed study": "Seeed Studio",
    "2D0": "Seeed studio",
    
    # Phonetic variations
    "seed city": "Seeed Studio",
    "seat city": "Seeed Studio",
    "seed street": "Seeed Studio",
    "seed study": "Seeed Studio",
    "sea to do": "Seeed Studio",
    "seed to do": "Seeed Studio",
    "si studio": "Seeed Studio",
    "seed store": "Seeed Studio",
    "seat store": "Seeed Studio",
    "seed stoodio": "Seeed Studio",
    "stood your":"Studio"
}

PRODUCT_CORRECTIONS = {
    # reSpeaker variations (voice product line)
    "re speaker": "reSpeaker",
    "re-speaker": "reSpeaker",
    "respeaker": "reSpeaker",
    "respeak": "reSpeaker",
    "re speaker array": "reSpeaker",
    "are speaker": "reSpeaker",
    "our speaker": "reSpeaker",
    "pre speaker": "reSpeaker",
    "free speaker": "reSpeaker",
    "real speaker": "reSpeaker",
    "resource speaker": "reSpeaker",
    "we speaker": "reSpeaker",
    "the speaker": "reSpeaker",
    "speaker array": "reSpeaker",
    "resp eaker": "reSpeaker",
    
    # SenseCraft variations (AI platform)
    "sense craft": "SenseCraft",
    "sensecraft": "SenseCraft",
    "cents craft": "SenseCraft",
    "sense graph": "SenseCraft",
    "sent craft": "SenseCraft",
    "sense craft ai": "SenseCraft",
    "sense card": "SenseCraft",
    "sen's craft": "SenseCraft",
    "census craft": "SenseCraft",
    "sense graph": "SenseCraft",
    
    # AI cameras
    "ai cam": "AI cameras",
    "a i camera": "AI cameras",
    "i camera": "AI cameras",
    "ai come": "AI cameras",
    "ai camas": "AI cameras",
    "ai came": "AI cameras",
    "ai cam ra": "AI cameras",
    
    # Fusion service
    "fusion services": "Fusion service",
    "fusions service": "Fusion service",
    "fusion server": "Fusion service",
    "fusing service": "Fusion service",
    "infusion service": "Fusion service",
}

TECHNICAL_CORRECTIONS = {
    # Vision/Voice confusion fixes (CRITICAL - most common STT errors)
    "mission": "vision",
    "missions": "vision",
    "mission solution": "vision solutions",
    "mission solutions": "vision solutions",
    "beijing": "vision",
    "his voice": "voice",
    "his voice solution": "voice solutions",
    "his voice solutions": "voice solutions",
    "boys": "voice",
    "boys solution": "voice solutions",
    "boys solutions": "voice solutions",
    "voice solution": "voice solutions",
    "voice saloons": "voice solutions",
    "voice illusion": "voice solutions",
    "vision solution": "vision solutions",
    "vision saloon": "vision solutions",
    "visual solutions": "vision solutions",
    "vision selection": "vision solutions",
    "visions": "vision solutions",
    
    # What is variations (common STT error at start)
    "that is": "what is",
    "what's": "what is",
    "what does": "what does",
    "watson": "what is",
    "but is":"what is",
    
    # Introduce variations
    "introduce yourself": "introduce this demo",
    "who are you": "introduce this demo",
    
    # Microphone/mic variations
    "mike": "mic",
    "mikes": "mics",
    "mike array": "mic array",
    "mic arra": "mic array",
    "micary": "mic array",
    "microphone array": "mic array",
    "my caray": "mic array",
    "mic arrays": "mic arrays",
    "microphone": "mic",
    "microphones": "mics",
    
    # Numeric mic arrays
    "2 mike": "2-mic",
    "2 mic": "2-mic",
    "two mic": "2-mic",
    "to mic": "2-mic",
    "to mike": "2-mic",
    "too mic": "2-mic",
    "2mike": "2-mic",
    "2mic": "2-mic",
    
    "4 mike": "4-mic",
    "4 mic": "4-mic",
    "for mic": "4-mic",
    "for mike": "4-mic",
    "four mic": "4-mic",
    "4mike": "4-mic",
    "4mic": "4-mic",
    "form ic": "4-mic",
    
    # NVIDIA Jetson variations
    "jetson": "NVIDIA Jetson",
    "jettison": "NVIDIA Jetson",
    "jet sum": "NVIDIA Jetson",
    "jet stone": "NVIDIA Jetson",
    "jesson": "NVIDIA Jetson",
    "jetsons": "NVIDIA Jetson",
    "nvidia jets on": "NVIDIA Jetson",
    "nvidia jet stone": "NVIDIA Jetson",
    "nvidia jetson": "NVIDIA Jetson",
    "jet son": "NVIDIA Jetson",
    "jets on": "NVIDIA Jetson",
    
    # Raspberry Pi variations
    "raspberry pie": "Raspberry Pi",
    "raspberry by": "Raspberry Pi",
    "raspberry pint": "Raspberry Pi",
    "rasp berry pi": "Raspberry Pi",
    "raspberry pi": "Raspberry Pi",
    "raspberry": "Raspberry Pi",
    "rasberry pi": "Raspberry Pi",
    "razberry pi": "Raspberry Pi",
    "pi board": "Raspberry Pi",
    "raspberry pie board": "Raspberry Pi",
    
    # AI algorithms
    "aec": "AEC",
    "a e c": "AEC",
    "a.e.c": "AEC",
    "doa": "DoA",
    "d o a": "DoA",
    "d.o.a": "DoA",
    "beam forming": "beamforming",
    "beam forming algorithm": "beamforming",
    "beamform": "beamforming",
    "eco cancellation": "echo cancellation",
    "echo cancel": "echo cancellation",
    "echo cancelation": "echo cancellation",
    "acoustic echo cancellation": "AEC",
    
    # Edge computing
    "edge computer": "edge computing",
    "edge computers": "edge computing",
    "edge commuter": "edge computing",
    "edge competing": "edge computing",
    "at computing": "edge computing",
    "h computing": "edge computing",
    
    # Solutions categories
    "sensor network": "sensor networks",
    "sensor network solution": "sensor network solutions",
    "sensor networks solution": "sensor network solutions",
    "sensor net work": "sensor networks",
    "sensor next work": "sensor networks",
    "censor networks": "sensor networks",
    "sensor solution": "sensor network solutions",
    
    "robotic solution": "robotics solutions",
    "robot solution": "robotics solutions",
    "robotic saloon": "robotics solutions",
    "robotic selection": "robotics solutions",
    "robot solutions": "robotics solutions",

    
    # Hardware components
    "lidar": "LiDAR",
    "lie dar": "LiDAR",
    "lighter": "LiDAR",
    "leader": "LiDAR",
    
    "imu": "IMU",
    "i m u": "IMU",
    "i.m.u": "IMU",
    
    "servo": "servos",
    "surface": "servos",
    
    # Demo variations
    "damo": "demo",
    "demmo": "demo",
    "demoah": "demo",
    "dem o": "demo",
    "demu": "demo",
    "teemo": "demo",
    "deemo": "demo",
    "dome oh": "demo",
    "them o": "demo",
    "dumbo": "demo",
    "dimmer": "demo",
    "day more": "demo",
    "memo": "demo",
    "limo": "demo",
    "dema": "demo",
    "tema": "demo",
    
    # Technical terms
    "tops": "TOPS",
    "top s": "TOPS",
    "open source": "open-source",
    "opensource": "open-source",
    "no code": "no-code",
    "nocode": "no-code",
}

# Combine all corrections
ALL_CORRECTIONS = {
    **BRAND_CORRECTIONS,
    **PRODUCT_CORRECTIONS,
    **TECHNICAL_CORRECTIONS
}


# ============================================
# FUZZY MATCHING FOR PARTIAL CORRECTIONS
# ============================================

def fuzzy_match(word, candidates, threshold=0.8):
    """
    Find the best fuzzy match for a word from candidates.
    Returns corrected word if similarity > threshold, else original.
    """
    word_lower = word.lower()
    best_match = None
    best_ratio = 0
    
    for candidate in candidates:
        ratio = SequenceMatcher(None, word_lower, candidate.lower()).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_match = candidate
    
    return best_match if best_match else word


# Key product/brand terms for fuzzy matching
FUZZY_KEYWORDS = [
    "Seeed", "Seeed Studio", "reSpeaker", "SenseCraft",
    "Jetson", "Raspberry Pi", "microphone", "camera",
    "sensor", "robotics", "vision", "voice", "LiDAR", "IMU"
]


# ============================================
# MAIN CORRECTION FUNCTIONS
# ============================================

def correct_stt_text(text: str, use_llm: bool = True) -> str:
    """
    Correct STT misrecognitions using multiple strategies:
    1. Direct dictionary replacement (300+ rules)
    2. Case normalization for product terms
    3. Fuzzy matching for partial matches
    4. LLM-based contextual correction (recommended)
    
    Args:
        text: Raw STT output text
        use_llm: Whether to use LLM for additional correction (default: True)
    
    Returns:
        Corrected text
    """
    if not text:
        return text
    
    corrected = text
    
    # Strategy 1: Direct replacements (case-insensitive)
    for wrong, correct in ALL_CORRECTIONS.items():
        # Use word boundaries to avoid partial replacements
        pattern = r'\b' + re.escape(wrong) + r'\b'
        corrected = re.sub(pattern, correct, corrected, flags=re.IGNORECASE)
    
    # Strategy 2: Case normalization for key product terms
    case_fixes = {
        "vision solutions": "Vision Solutions",
        "voice solutions": "Voice Solutions",
        "sensor network solutions": "Sensor Network Solutions",
        "robotics solutions": "Robotics Solutions",
    }
    
    for wrong_case, correct_case in case_fixes.items():
        pattern = r'\b' + re.escape(wrong_case) + r'\b'
        corrected = re.sub(pattern, correct_case, corrected, flags=re.IGNORECASE)
    
    # Strategy 3: Fuzzy matching for unmatched words
    words = corrected.split()
    corrected_words = []
    
    for word in words:
        # Skip short words and already corrected terms
        if len(word) <= 3 or word in ["Seeed", "reSpeaker", "SenseCraft"]:
            corrected_words.append(word)
            continue
        
        # Try fuzzy matching
        fuzzy_corrected = fuzzy_match(word, FUZZY_KEYWORDS, threshold=0.75)
        corrected_words.append(fuzzy_corrected)
    
    corrected = " ".join(corrected_words)
    
    # Strategy 4: LLM-based correction (contextual understanding)
    if use_llm:
        corrected = llm_correct_text(corrected)
    
    return corrected


def llm_correct_text(text: str) -> str:
    """
    Use LLM to correct technical terminology in context.
    CRITICAL: This should ONLY fix obvious typos, not rewrite queries!
    """
    try:
        from ollama import chat
        
        # IMPROVED PROMPT: Much stricter instructions
        prompt = f"""You are a minimal text correction tool. Your ONLY job is fixing brand/product name typos.

Input: "{text}"

Rules (CRITICAL - DO NOT VIOLATE):
1. ONLY fix these exact typos if present:
   - "seed" or "seat" → "Seeed"
   - "stood"→ "Studio"
   - "mission" → "vision"
   - "ice" → "voice"
   - "botic" → "robotic"
   
2. DO NOT change:
   - Question structure
   - Word order
   - Sentence meaning
   - Any words that aren't in the typo list above

3. If no typos from list #1 are found, return EXACTLY the input text unchanged

4. Output ONLY the corrected text with no quotes, no explanation, no preamble

Corrected text:"""

        response = chat(
            model="gemma3:4b",
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,  # Very low creativity
                "top_p": 0.5,        # More deterministic
            }
        )
        
        corrected = response.message.content.strip()
        
        # Remove quotes if LLM adds them
        corrected = corrected.strip('"').strip("'").strip()
        
        # Safety check: If LLM output is too different, reject it
        if len(corrected) > len(text) * 1.5 or len(corrected) < len(text) * 0.5:
            print(f"[LLM] Output too different, rejecting: '{corrected}'")
            return text
        
        return corrected
        
    except Exception as e:
        print(f"[LLM Correction Error] {e}")
        return text


# ============================================
# UTILITY FUNCTIONS
# ============================================

def extract_keywords(text: str) -> list:
    """
    Extract key product/brand terms from corrected text.
    Useful for debugging and understanding query intent.
    """
    keywords = []
    text_lower = text.lower()
    
    search_terms = [
        "seeed", "respeaker", "sensecraft", "jetson", "raspberry pi",
        "mic array", "camera", "sensor", "robotics", "vision", "voice",
        "fusion", "aec", "beamforming", "lidar", "imu", "tops"
    ]
    
    for term in search_terms:
        if term in text_lower:
            keywords.append(term)
    
    return keywords


def get_correction_stats(original: str, corrected: str) -> dict:
    """
    Get statistics about what was corrected.
    Useful for monitoring and improvement.
    """
    return {
        "original": original,
        "corrected": corrected,
        "changed": original != corrected,
        "keywords_found": extract_keywords(corrected),
        "length_original": len(original),
        "length_corrected": len(corrected)
    }


# ============================================
# TESTING EXAMPLES
# ============================================

if __name__ == "__main__":
    test_cases = [
        # Brand variations
        "What is seat studio?",
        "Tell me about c studio products",
        "Does seeds studio have cameras?",
        "What is c2d-o?",
        
        # Product names
        "Tell me about re speaker",
        "What is sense craft platform?",
        "Show me are speaker arrays",
        
        # Vision/Voice confusion (CRITICAL TESTS)
        "What are seed mission solutions?",
        "Tell me about his voice solutions",
        "What is seed studio mission products?",
        "Tell me about boys solutions",
        "What are seed missions?",
        "Does seed have his voice arrays?",
        
        # Technical terms
        "Does seed have 4 mike arrays?",
        "What cameras work with raspberry pie?",
        "Tell me about jetson support",
        
        # Solutions
        "What are seed voice solution?",
        "Tell me about vision saloon",
        "What robotic solution do you offer?",
        
        # Complex queries
        "Can I use seat studio re speaker with raspberry pie?",
        "What are the specs of the 4 mike array?",
        "Does c studio support nvidia jets on?",
        "Tell me about c2d-o mission solutions",
    ]
    
    print("=" * 80)
    print("STT CORRECTION TESTING (WITH LLM)")
    print("=" * 80)
    
    for test in test_cases:
        corrected = correct_stt_text(test, use_llm=True)
        print(f"\nOriginal:  {test}")
        print(f"Corrected: {corrected}")
        print(f"Keywords:  {extract_keywords(corrected)}")
        print("-" * 80)