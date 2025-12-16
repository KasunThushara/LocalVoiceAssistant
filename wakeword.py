import speech_recognition as sr
import os
import sys
import requests
import io
import json
import wave
import threading
import time
import re
from collections import deque
import simpleaudio as sa

# Suppress ALSA errors
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
if sys.platform == 'linux':
    try:
        from ctypes import *
        ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        def py_error_handler(filename, line, function, err, fmt):
            pass
        c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        asound = cdll.LoadLibrary('libasound.so.2')
        asound.snd_lib_error_set_handler(c_error_handler)
    except:
        pass

# Configuration
WAKEWORD = "speaker"
RECORD_DURATION = 5
WHISPER_URL = "http://127.0.0.1:8080/inference"
LISTEN_PHRASE_LIMIT = 3
WHISPER_TIMEOUT = 1
STOP_MODE_TIMEOUT = 1
STOP_MODE_PHRASE_LIMIT = 1.5

# ============================================
# SMART STOP WORD DETECTION
# ============================================

PRIMARY_STOP_WORDS = ["stop", "no"]
STT_MISHEARINGS_STOP = ["top", "stock", "stap", "stopped", "shop", "cop", "stope", "step", "spot"]
STT_MISHEARINGS_NO = ["nah", "nope"]

def is_stop_command(text: str) -> bool:
    """
    Smart stop detection - only triggers on actual stop commands.
    
    Will trigger on:
    - "stop", "no" (exact)
    - "top", "shop" (STT mishearings of "stop")
    - "stop talking", "no thanks" (short stop phrases)
    
    Will NOT trigger on:
    - "I don't know" (has "know" but not a stop)
    - "show me cameras" (has "show" but not a stop)
    - "what's new" (has "new" but not a stop)
    """
    if not text:
        return False
    
    # Clean text
    text_clean = text.lower().strip()
    text_clean = re.sub(r'[^\w\s]', '', text_clean)  # Remove punctuation
    text_clean = text_clean.strip()
    
    if not text_clean or len(text_clean) < 2:
        return False
    
    words = text_clean.split()
    
    # Rule 1: Exact match "stop" or "no"
    if text_clean in PRIMARY_STOP_WORDS:
        return True
    
    # Rule 2: Known "stop" mishearings (standalone)
    if text_clean in STT_MISHEARINGS_STOP:
        return True
    
    # Rule 3: Short phrases starting with stop word (≤3 words)
    if len(words) <= 3:
        first_word = words[0]
        if first_word in PRIMARY_STOP_WORDS or first_word in STT_MISHEARINGS_STOP:
            return True
    
    # Rule 4: Standalone "no" variants
    if len(words) <= 2:
        if text_clean in ["no", "nah", "nope"]:
            return True
        # "no thanks", "no stop"
        if words[0] == "no":
            return True
    
    # Rule 5: Starts with "stop" or "top"
    if text_clean.startswith("stop ") or text_clean.startswith("top "):
        return True
    
    return False


# ============================================
# EXISTING WHISPER/AUDIO CODE
# ============================================

# Optimization: Cache recent transcriptions
recent_transcriptions = deque(maxlen=5)
transcription_cache_lock = threading.Lock()

r = sr.Recognizer()
r.energy_threshold = 2500
r.dynamic_energy_threshold = True
r.pause_threshold = 0.5
r.non_speaking_duration = 0.4

# Global state for stop monitoring
stop_monitoring_active = False
stop_callback_func = None
stop_monitor_lock = threading.Lock()


def play_sound(path):
    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        wave_obj.play()
    except Exception as e:
        print(f"[⚠️ AUDIO ERROR] Could not play {path}: {e}")


def transcribe_with_whisper(audio_data, fast_mode=False):
    """Transcribe audio using local Whisper server."""
    try:
        wav_data = audio_data.get_wav_data()
        
        # Check cache
        audio_hash = hash(wav_data[:1000])
        with transcription_cache_lock:
            if audio_hash in recent_transcriptions:
                return None
        
        files = {
            'file': ('audio.wav', io.BytesIO(wav_data), 'audio/wav')
        }
        
        if fast_mode:
            data = {'temperature': '0.0', 'response_format': 'json'}
            timeout = 0.8
        else:
            data = {'temperature': '0.0', 'temperature_inc': '0.2', 'response_format': 'json'}
            timeout = WHISPER_TIMEOUT
        
        response = requests.post(WHISPER_URL, files=files, data=data, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('text', '').strip()
            
            with transcription_cache_lock:
                recent_transcriptions.append(audio_hash)
            
            return text
        else:
            return None
            
    except requests.exceptions.Timeout:
        return None
    except Exception as e:
        if not fast_mode:
            print(f"[⚠️] Whisper error: {e}")
        return None


def record_audio(filename="record.wav"):
    """Record audio and save to file."""
    with sr.Microphone(sample_rate=16000) as source:
        print(f"[🎙️] Recording for {RECORD_DURATION} seconds...")
        audio = r.record(source, duration=RECORD_DURATION)
    
    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())
    
    print(f"[✅] Saved: {filename}")
    return filename


def wakeword_listener(process_callback, status_callback, stop_callback):
    """
    Single-thread listener with smart stop detection.
    """
    global stop_monitoring_active, stop_callback_func
    
    stop_callback_func = stop_callback
    
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        print("[🔊] Wake word listener ready. Say 'hi speaker'...")
        print(f"[ℹ️] Using Whisper at {WHISPER_URL}")
        print(f"[🛑] Stop commands: 'stop' or 'no'")
        print(f"[⚡] Smart stop detection enabled")

    consecutive_errors = 0
    max_errors = 5
    stop_checks = 0

    while True:
        try:
            with stop_monitor_lock:
                monitoring_stop = stop_monitoring_active
            
            if monitoring_stop:
                phrase_limit = STOP_MODE_PHRASE_LIMIT
                timeout = STOP_MODE_TIMEOUT
                stop_checks += 1
                if stop_checks % 5 == 0:
                    print(f"[🛑] Stop detection active (check #{stop_checks})...")
            else:
                phrase_limit = LISTEN_PHRASE_LIMIT
                timeout = 3.0
                stop_checks = 0
            
            # Listen
            with sr.Microphone() as source:
                try:
                    audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
                except sr.WaitTimeoutError:
                    continue
            
            # Transcribe
            text = transcribe_with_whisper(audio, fast_mode=monitoring_stop)
            
            if text is None:
                continue
            
            text_lower = text.lower()
            consecutive_errors = 0
            
            with stop_monitor_lock:
                monitoring_stop = stop_monitoring_active
            
            if monitoring_stop:
                # STOP DETECTION MODE
                if text_lower and not text_lower.startswith('[blank'):
                    print(f"[👂] Stop mode: '{text_lower}'")
                
                # Use smart detection
                if is_stop_command(text_lower):
                    play_sound("media/stop.wav")
                    print("[🛑] ╔═══ STOP COMMAND DETECTED ═══╗")
                    print(f"[🛑] Detected: '{text_lower}'")
                    print("[🛑] ╚═══════════════════════════╝")
                    
                    if stop_callback_func:
                        stop_callback_func()
                    
                    with stop_monitor_lock:
                        stop_monitoring_active = False
                    
                    print("[✓] Returning to wake word mode")
                    stop_checks = 0
                    continue
                    
            else:
                # WAKE WORD MODE
                print(f"[👂] Wake word mode: '{text_lower}'")
                
                if WAKEWORD in text_lower:
                    play_sound("media/wake.wav")
                    print("[🚀] Wake word detected!")
                    status_callback("🚀 Wake word detected!")
                    
                    status_callback(f"🎙️ Recording for {RECORD_DURATION} seconds...")
                    file_path = record_audio()
                    
                    # Enable stop monitoring
                    with stop_monitor_lock:
                        stop_monitoring_active = True
                    print("[🛑] Stop detection ACTIVATED")
                    stop_checks = 0
                    
                    process_callback(file_path)

        except Exception as e:
            consecutive_errors += 1
            print(f"[⚠️] Error ({consecutive_errors}/{max_errors}): {e}")
            
            if consecutive_errors >= max_errors:
                print(f"[❌] Too many errors, stopping listener")
                status_callback("❌ Listener stopped due to errors")
                break
            
            time.sleep(0.1)
            continue


def test_whisper_connection():
    """Test if Whisper server is available."""
    try:
        response = requests.get(WHISPER_URL.replace('/inference', '/'), timeout=2)
        print("[✅] Whisper server is reachable")
        return True
    except:
        print("[❌] Whisper server not reachable at", WHISPER_URL)
        print("    Please start the Whisper server first!")
        return False


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING STOP DETECTION")
    print("=" * 60)
    
    test_phrases = [
        ("stop", True),
        ("no", True),
        ("stop talking", True),
        ("top", True),
        ("no thanks", True),
        ("I don't know", False),
        ("show me cameras", False),
        ("what do you know about Seeed", False),
        ("tell me about the workshop", False),
    ]
    
    for phrase, expected in test_phrases:
        result = is_stop_command(phrase)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{phrase:30}' → Stop={result} (expected={expected})")
    
    print("=" * 60 + "\n")