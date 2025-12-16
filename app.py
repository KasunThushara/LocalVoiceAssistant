from flask import Flask, render_template, jsonify
import threading
import requests
from wakeword import wakeword_listener
from ollama import chat
from rag import retrieve_relevant, build_rag_prompt, build_general_prompt, init_rag
from stt_corrector import correct_stt_text, get_correction_stats
import re
import wave
import os
import json
from piper import PiperVoice

import logging

# Suppress Flask dev server logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

PIPER_MODEL_PATH = "models/en_US-amy-low.onnx"
TTS_OUTPUT_PATH = "static/output.wav"

state = {
    "status": "🎧 Listening for wake word...",
    "stt_text": "",
    "stt_corrected": "",
    "llm_response": "",
    "tts_ready": False,
    "rag_used": False,
    "stop_requested": False
}

lock = threading.Lock()


# -----------------------------------------------------
# CLEAN TEXT
# -----------------------------------------------------
def clean_text(text):
    text = re.sub(r"\*+", "", text)
    
    # First, normalize different types of apostrophes to the standard ASCII apostrophe
    text = text.replace("'", "'")  # curly single quotes to straight
    text = text.replace(""", "'")  # left double quote to apostrophe (if needed)
    text = text.replace(""", "'")  # right double quote to apostrophe (if needed)
    
    # Replace curly apostrophe (U+2019) with straight apostrophe
    text = text.replace("’", "'")
    
    # Now keep ASCII apostrophes when filtering
    # Allow ASCII characters plus the apostrophe (39)
    text = re.sub(r"[^\x00-\x7F']+", " ", text)
    
    return re.sub(r"\s+", " ", text).strip()


# -----------------------------------------------------
# GENERATE TTS (Piper)
# -----------------------------------------------------
def generate_tts(text: str):
    try:
        voice = PiperVoice.load(PIPER_MODEL_PATH)
        with wave.open(TTS_OUTPUT_PATH, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        return True
    except Exception as e:
        print("[TTS ERROR]", e)
        return False


# -----------------------------------------------------
# MAIN PIPELINE: WHISPER → CORRECTION → RAG → OLLAMA → TTS
# -----------------------------------------------------
def process_audio(file_path):
    with lock:
        state["status"] = "📡 Sending to Whisper..."
        # IMPORTANT: Don't clear these yet - wait for correction
        state["llm_response"] = ""
        state["tts_ready"] = False
        state["rag_used"] = False
        state["stop_requested"] = False

    # -------------------------------
    # Step 1: Whisper STT
    # -------------------------------
    raw_stt_text = ""  # Store temporarily, don't expose to UI yet
    
    try:
        files = {"file": open(file_path, "rb")}
        data = {"temperature": "0.0", "response_format": "json"}

        resp = requests.post("http://127.0.0.1:8080/inference",
                             files=files, data=data, timeout=10)

        try:
            whisper_json = json.loads(resp.text)
            raw_stt_text = clean_text(whisper_json.get("text", resp.text))
        except json.JSONDecodeError:
            raw_stt_text = clean_text(resp.text)

        # Don't update state["stt_text"] yet - wait for correction
        print(f"[RAW STT] {raw_stt_text}")
        
        with lock:
            state["status"] = "🔧 Correcting terminology..."

    except Exception as e:
        with lock:
            state["stt_text"] = f"Whisper error: {e}"
            state["stt_corrected"] = ""
            state["status"] = "🎧 Listening for wake word..."
        return

    # -------------------------------
    # Step 2: STT CORRECTION
    # -------------------------------
    try:
        corrected_text = correct_stt_text(raw_stt_text, use_llm=True)
        
        # NOW update both at the same time
        with lock:
            state["stt_text"] = raw_stt_text  # For debug purposes
            state["stt_corrected"] = corrected_text  # This is what UI should show
            state["status"] = "🔍 Checking product database..."
        
        stats = get_correction_stats(raw_stt_text, corrected_text)
        if stats["changed"]:
            print(f"[STT CORRECTED] '{raw_stt_text}' → '{corrected_text}'")
            print(f"[KEYWORDS] {stats['keywords_found']}")
        else:
            print(f"[NO CORRECTION NEEDED] '{corrected_text}'")
        
    except Exception as e:
        print(f"[CORRECTION ERROR] {e}")
        corrected_text = raw_stt_text
        with lock:
            state["stt_text"] = raw_stt_text
            state["stt_corrected"] = corrected_text

    # -------------------------------
    # Step 3: RAG Retrieval
    # -------------------------------
    try:
        context, is_relevant = retrieve_relevant(corrected_text, k=3)
        
        with lock:
            state["rag_used"] = is_relevant
            if is_relevant:
                state["status"] = "💡 Generating response with product info..."
            else:
                state["status"] = "💡 Generating general response..."
    except Exception as e:
        print(f"[RAG ERROR] {e}")
        context, is_relevant = "", False

    # -------------------------------
    # Step 4: LLM Response (Ollama)
    # -------------------------------
    try:
        if is_relevant:
            prompt = build_rag_prompt(corrected_text, context)
        else:
            prompt = build_general_prompt(corrected_text)

        llm_resp = chat(
            model="gemma3:4b",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = clean_text(llm_resp.message.content)

        with lock:
            state["llm_response"] = answer

    except Exception as e:
        with lock:
            state["llm_response"] = f"LLM error: {e}"

    # -------------------------------
    # Step 5: TTS Generation
    # -------------------------------
    ok = generate_tts(state["llm_response"])

    with lock:
        state["tts_ready"] = ok
        state["status"] = "🔊 Playing response..."


# -----------------------------------------------------
# STOP DETECTION CALLBACK
# -----------------------------------------------------
def handle_stop_command():
    """Called when stop command is detected during TTS playback."""
    with lock:
        state["stop_requested"] = True
        state["status"] = "🛑 Stopped by voice command"
        state["tts_ready"] = False
    
    # Disable stop monitoring immediately
    import wakeword
    with wakeword.stop_monitor_lock:
        wakeword.stop_monitoring_active = False
    
    print("[🛑] Stop command received - audio will be interrupted!")


# -----------------------------------------------------
# ROUTES
# -----------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    with lock:
        return jsonify(state)


@app.route("/reset_after_tts", methods=["POST"])
def reset_after_tts():
    """Called when TTS finishes playing or is stopped."""
    from wakeword import stop_monitor_lock, stop_monitoring_active
    
    with lock:
        state["status"] = "🎧 Listening for wake word..."
        state["tts_ready"] = False
        state["stop_requested"] = False
    
    # Disable stop monitoring when returning to wake word mode
    import wakeword
    with wakeword.stop_monitor_lock:
        wakeword.stop_monitoring_active = False
    
    return jsonify({"success": True})


@app.route("/stop_tts", methods=["POST"])
def stop_tts():
    """Manual stop button endpoint."""
    with lock:
        state["stop_requested"] = True
        state["tts_ready"] = False
        state["status"] = "🛑 Stopped manually"
    print("[🛑] Manual stop requested")
    return jsonify({"success": True})


# -----------------------------------------------------
# STARTUP
# -----------------------------------------------------
def update_status(msg):
    with lock:
        state["status"] = msg


if __name__ == "__main__":
    print("=" * 60)
    print("🎤 WAKE WORD AI SYSTEM STARTING...")
    print("=" * 60)

    # Initialize RAG system
    init_rag()

    thread = threading.Thread(
        target=wakeword_listener,
        args=(process_audio, update_status, handle_stop_command),
        daemon=True
    )
    thread.start()

    print("✓ Wake word listener running")
    print("✓ RAG system loaded")
    print("✓ STT correction enabled")
    print("✓ Stop command detection enabled")
    print("✓ Web interface: http://localhost:5000")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)