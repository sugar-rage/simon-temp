"""
Diagnostic test for SpeechInput dependencies.
Tests each component (Whisper, sounddevice, webrtcvad) separately.
"""

print("=" * 60)
print("  SIMON Speech Input — Dependency Check")
print("=" * 60)

# --- 1. Check sounddevice ---
print("\n[1/4] Checking sounddevice...")
try:
    import sounddevice as sd
    devices = sd.query_devices()
    print(f"  [OK] sounddevice OK - Found {len(devices)} audio device(s)")
    # Show default input device
    try:
        default_in = sd.query_devices(kind='input')
        print(f"  Default input: {default_in['name']}")
    except Exception as e:
        print(f"  [WARN] No default input device: {e}")
except ImportError:
    print("  [FAIL] sounddevice NOT installed - run: pip install sounddevice")
except Exception as e:
    print(f"  [FAIL] sounddevice error: {e}")

# --- 2. Check whisper ---
print("\n[2/4] Checking openai-whisper...")
try:
    import whisper
    print(f"  [OK] whisper imported OK")
    print("  Loading 'base' model (this may take a moment)...")
    model = whisper.load_model("base")
    print(f"  [OK] Whisper 'base' model loaded OK")
except ImportError:
    print("  [FAIL] whisper NOT installed - run: pip install openai-whisper")
except Exception as e:
    print(f"  [FAIL] whisper error: {e}")

# --- 3. Check webrtcvad (optional) ---
print("\n[3/4] Checking webrtcvad (optional)...")
try:
    import webrtcvad
    vad = webrtcvad.Vad(2)
    print("  [OK] webrtcvad OK")
except ImportError:
    print("  [WARN] webrtcvad not installed (optional) - will use timer-based recording")
except Exception as e:
    print(f"  [FAIL] webrtcvad error: {e}")

# --- 4. Check scipy (needed for transcription) ---
print("\n[4/4] Checking scipy...")
try:
    from scipy.io.wavfile import write
    print("  [OK] scipy OK")
except ImportError:
    print("  [FAIL] scipy NOT installed - run: pip install scipy")
except Exception as e:
    print(f"  [FAIL] scipy error: {e}")

# --- 5. Quick microphone test ---
print("\n[Mic Test] Finding a working microphone...")
try:
    import numpy as np
    import sounddevice as sd

    devices = sd.query_devices()
    host_apis = sd.query_hostapis()

    # Skip loopback/speaker devices
    loopback_kw = ["speaker", "loopback", "stereo mix", "pc speaker",
                   "output", "headphone"]

    def is_real_mic(name):
        return not any(kw in name.lower() for kw in loopback_kw)

    # Prefer WDM-KS > WASAPI > DirectSound > MME
    preferred = ["Windows WDM-KS", "Windows WASAPI",
                 "Windows DirectSound", "MME"]
    real_mics = []
    loopback_devs = []
    for api_name in preferred:
        for api in host_apis:
            if api["name"] == api_name:
                for did in api["devices"]:
                    d = devices[did]
                    if d["max_input_channels"] > 0:
                        entry = (did, d["name"], int(d["default_samplerate"]))
                        if is_real_mic(d["name"]):
                            real_mics.append(entry)
                        else:
                            loopback_devs.append(entry)

    mic_ok = False
    for dev_id, dev_name, rate in real_mics + loopback_devs:
        try:
            audio = sd.rec(int(2 * rate), samplerate=rate,
                           channels=1, dtype="int16", device=dev_id)
            sd.wait()
            max_vol = np.max(np.abs(audio.astype(float)))
            print(f"  [OK] Mic OK - [{dev_id}] {dev_name} @ {rate}Hz, max vol: {max_vol:.0f}")
            if max_vol < 100:
                print("  [WARN] Very low volume - mic might be muted")
            else:
                print("  [OK] Mic is picking up audio")
            mic_ok = True
            break
        except Exception:
            continue

    if not mic_ok:
        print("  [FAIL] No working microphone found across any audio backend!")
except Exception as e:
    print(f"  [FAIL] Mic test failed: {e}")

print("\n" + "=" * 60)
print("  Diagnostic complete!")
print("=" * 60)
