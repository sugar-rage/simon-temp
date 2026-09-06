"""
SIMON — Smart Intelligent Mobility & Outdoor Navigator
Main Application Entry Point

Integrates:
  - perception/merger.py     → Camera frame analysis (objects, faces, OCR)
  - logic/logic_controller   → Decision making (spatial, priority, safety)
  - logic/navigation_logic   → OSM-based outdoor navigation
  - speech/speech_output     → Text-to-speech output (pyttsx3)
  - speech/speech_input      → Whisper-based voice commands
"""

# pyrefly: ignore [missing-import]
import cv2
import time
import logging
from perception.merger import PerceptionMerger
from logic.logic_controller import LogicController
from logic.navigation_logic import NavigationLogic
from logic.detection_schema import Detection
from speech._legacy.speech_output import VoiceEngine
from speech._legacy.speech_input import SpeechInput

FRAME_W, FRAME_H = 640, 480
logger = logging.getLogger("SIMON")


def main():
    import threading
    shutdown_event = threading.Event()

    # --- 1. Fast, Core Functionality First ---
    tts = VoiceEngine()
    stt = SpeechInput(voice_engine=tts)
    nav = NavigationLogic(voice_engine=tts)
    stt.start()

    # Wire barge-in control: VoiceEngine will automatically call
    # stt.pause_for_tts() / stt.resume_after_tts() around each utterance.
    tts.register_speech_input(stt)

    print("SIMON running with Perception Merger (Objects + Faces + OCR)...")
    print("Press V to toggle voice commands, N to navigate, Q/ESC to quit.")
    if nav.is_available:
        print("Navigation module ready (OSM routing).")
    else:
        print("Navigation unavailable — install: pip install osmnx networkx geopy")

    # --- 2. Slower Perception Modules Second ---
    merger = PerceptionMerger(shutdown_event=shutdown_event)
    logic = LogicController(FRAME_W, FRAME_H)

    # --- 3. Async Device Discovery (Camera) ---
    import threading
    import numpy as np
    cap = None
    camera_lock = threading.Lock()
    
    def init_camera():
        nonlocal cap
        while not shutdown_event.is_set():
            with camera_lock:
                is_open = cap is not None and cap.isOpened()
            
            if not is_open:
                try:
                    temp_cap = cv2.VideoCapture(0)
                    if temp_cap.isOpened():
                        with camera_lock:
                            cap = temp_cap
                except Exception as e:
                    logger.error(f"[Camera] Device scan exception: {e}")
            
            time.sleep(1.0) # Check camera health / reconnect periodically

    camera_thread = threading.Thread(target=init_camera, daemon=True)
    camera_thread.start()

    # Face greeting cooldown


    try:
        while True:
            with camera_lock:
                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                else:
                    ret = False

            if not ret:
                # Placeholder frame while discovering devices or if no device found
                frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
                cv2.putText(frame, "Discovering Camera Device...", (120, FRAME_H//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                time.sleep(0.03)  # Reduce CPU spin
            else:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            # --- Run unified perception pipeline ---
            try:
                result = merger.process(frame)
            except Exception:
                logger.error("[Perception] Frame processing failed", exc_info=True)
                continue

            # --- 1. Unified Perception → Logic Controller ---
            all_detections = result.detections + result.faces + result.ocr
            if all_detections:
                try:
                    res = logic.process(all_detections)
                    if res:
                        speech_text, priority = res
                        tts.speak(speech_text, priority=priority)
                except Exception:
                    logger.error("[Logic] Detection processing failed", exc_info=True)

            # --- 4. Handle voice commands ---
            voice_cmd = stt.get_command()
            if voice_cmd:
                action = voice_cmd["action"]
                args = voice_cmd.get("args", "")
                print(f"[VoiceCmd] {action} {args}")

                if action == "read_text" and merger.ocr_reader:
                    ocr_results = merger.ocr_reader.read_text(frame)
                    text = merger.ocr_reader.get_readable_text(ocr_results)
                    if text:
                        tts.speak(f"I can read: {text[:100]}")
                    else:
                        tts.speak("No text detected")
                elif action == "save_face":
                    unknown = result.unknown_faces
                    if unknown and merger.face_recognizer:
                        face = unknown[0]
                        if args:
                            merger.face_recognizer.save_face(frame, face.face_obj, args)
                            tts.speak(f"Saved face as {args}", priority=3)
                        else:
                            tts.speak("Please say the name after save face.", priority=3)
                    else:
                        tts.speak("No unknown face detected to save.", priority=3)
                elif action == "navigate":
                    if args:
                        nav.navigate_to(args)
                    else:
                        tts.speak("Where would you like to go?", priority=3)
                elif action == "cancel_nav":
                    nav.cancel()
                elif action == "status":
                    n_obj = len(result.detections)
                    n_face = len(result.faces)
                    nav_status = "navigating" if nav.is_navigating else "idle"
                    tts.speak(f"I see {n_obj} objects and {n_face} faces. Navigation: {nav_status}.", priority=3)
                elif action == "stop":
                    nav.cancel()
                    tts.speak("Goodbye.", priority=3)
                    break

            # --- 5. Handle keyboard input ---
            cv2.imshow("SIMON Vision", result.frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # q or ESC to quit
                break
            elif key == ord("v"):
                stt.toggle()
            elif key == ord("s"):
                # Save the first unknown face
                unknown = result.unknown_faces
                if unknown and merger.face_recognizer:
                    face = unknown[0]
                    name = input("Enter name for this face: ").strip()
                    if name:
                        merger.face_recognizer.save_face(frame, face.face_obj, name)
                        tts.speak(f"Saved face as {name}", priority=3)
            elif key == ord("r"):
                # Force read text from current frame
                if merger.ocr_reader:
                    ocr_results = merger.ocr_reader.read_text(frame)
                    text = merger.ocr_reader.get_readable_text(ocr_results)
                    if text:
                        print(f"[OCR] {text}")
                        tts.speak(f"I can read: {text[:100]}")
                    else:
                        tts.speak("No text detected")

    finally:
        # Clean Shutdown
        print("Initiating clean shutdown...")
        shutdown_event.set()
        
        stt.stop()
        tts.stop()
        merger.stop()
        
        with camera_lock:
            if cap is not None:
                cap.release()
                
        # Wait up to 2 seconds for camera thread to exit cleanly
        camera_thread.join(timeout=2.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()