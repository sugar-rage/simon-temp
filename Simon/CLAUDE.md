# SIMON — System Knowledge Base

> **For AI assistants only.** This is not human documentation. It is a persistent knowledge base optimized for fresh Claude sessions to immediately resume development on this project.

## Project Overview

**SIMON** (Smart Intelligent Mobility & Outdoor Navigator) is an offline-first AI-powered assistive system for visually impaired users. It perceives the world through a camera, reasons about what it sees, navigates the user safely, and communicates entirely through speech.

**The Speech subsystem is COMPLETE and FROZEN.** The `speech/` directory contains a production-grade, safety-critical speech subsystem (~80+ files, 274 tests). It is treated as a stable external dependency with public APIs. **Do NOT modify any speech files.**

**All system phases (1-6) are COMPLETE.** Core Infrastructure, Vision, Core Intelligence, Navigation, Integration, and Hardening — 337 tests passing. See `implementation_plan.md` for the complete architecture document.

**Design philosophy:**
- Accuracy > speed > simplicity
- Incorrect recognition can cause **safety hazards** (missed obstacles, wrong navigation)
- No shortcuts: heavy dependencies are acceptable if they improve robustness
- Constructor dependency injection everywhere; no global singletons
- Every component has an abstract base class for testability and swapability
- Fallback chains: every critical component has a lighter fallback (Silero→Energy VAD, OpenWakeWord→TextMatch, Neural TTS→pyttsx3)

**Engineering goals:**
- Recognition accuracy maximization through multi-engine STT, adaptive decoding, hallucination filtering, confidence gating
- Sub-200ms perceived TTS latency via sentence-level streaming synthesis
- Priority-based preemption: emergency warnings instantly interrupt low-priority speech
- Zero-downtime device hot-swap and automatic error recovery
- Structured logging and metrics collection on every pipeline stage

---

## Current Progress

### Speech Subsystem (FROZEN)

| Phase | Status | Description |
|-------|--------|-------------|
| **Speech 1** | ✅ Complete | Core Infrastructure + STT Pipeline (35 files, 85 tests) |
| **Speech 2** | ✅ Complete | TTS Pipeline + Speak Pipeline (11 files) |
| **Speech 3** | ✅ Complete | Multi-Engine STT + Scene Classification (13 files, 50 tests) |
| **Speech 4** | ✅ Complete | Context Memory + Safety + Vocabulary + Preprocessing (22 files, 90 tests) |
| **Speech 5** | ✅ Complete | Personal Adaptation + Speaker Verification + Dashboard + Integration (15 files, 49 tests) |

**Speech subsystem: 100% complete. 80+ files, 274 tests, 0 failures. DO NOT MODIFY.**

### System Architecture (ALL PHASES COMPLETE)

| Phase | Status | Description |
|-------|--------|-------------|
| **System 1** | ✅ Complete | Core Infrastructure — Event Bus, Models, Config, State Machine, Capabilities (26 files, 110 tests) |
| **System 2** | ✅ Complete | Vision Subsystem — Camera, YOLO, OCR, Faces, Perception Fusion (19 files, 41 tests) |
| **System 3** | ✅ Complete | Core Intelligence — Logic Controller, Safety, Context, Memory (8 files, 74 tests) |
| **System 4** | ✅ Complete | Navigation Subsystem — GPS, Routing, Guidance, Pipeline (7 files, 33 tests) |
| **System 5** | ✅ Complete | Integration & App — AppController, Task Planner, Actions, Plugins, Health, Resources (18 files, 63 tests) |
| **System 6** | ✅ Complete | Hardening — Benchmarks, Stress Tests, Memory Profiling (6 files, 28 tests) |

**Cumulative test results: 337 passed, 3 skipped, 0 failures.**

**Legacy code archived:**
- `perception/` → `perception/_legacy/` (replaced by `vision/`)
- `logic/` → `logic/_legacy/` (replaced by `core/logic/`, `core/safety/`, `core/context/`, `core/memory/`)

**System architecture: designed and frozen. See `implementation_plan.md` for full document.**

---

## Project Architecture

### `speech/models/`
**Responsibility:** Core data structures shared across all modules.
**Dependencies:** `numpy` only.
**Key interfaces:**
- `AudioFrame` — immutable-by-convention dataclass; universal audio currency between all pipeline stages. Has `data` (np.ndarray), `sample_rate`, `channels`, `dtype`, `timestamp`, `device_id`, `sequence_id`. Supports `.concatenate()`, `.to_float32()`, `.to_int16()`, `.resample()`, `.rms()`.
- `SpeechPriority` — IntEnum (EMERGENCY=1 → DEBUG=10). Has `.should_interrupt`, `.is_safety_critical`, `.from_legacy()`.
- `SpeechCommand` — Output of listen pipeline. Has `.action`, `.args`, `.confidence`, `.raw_transcript`, `.status`. Supports backward-compatible dict access (`cmd["action"]` works).
- `SceneType` — Enum for acoustic environments (QUIET_ROOM, OFFICE, STREET, etc.).
- `DecodingParams` — Per-scene STT decoding parameters (beam_size, temperature, patience, etc.).

### `speech/audio/`
**Responsibility:** Audio I/O, device management, DSP processing chain.
**Dependencies:** `sounddevice`, `numpy`, `scipy` (optional: `deepfilternet`).
**Key interfaces:**
- `DeviceManager` — Enumerates audio devices, ranks by quality (WDM-KS > DirectSound), monitors for hot-swap. `.get_best_device()` → `AudioDevice`.
- `AudioStream` — PortAudio callback-based input stream with ring buffer. `.start()`, `.stop()`, `.pause()`, `.resume()`, `.read_frame()`, `.read_frames()`.
- `DSPPipeline` — Chain-of-responsibility orchestrator. `.process(AudioFrame)` → `AudioFrame`. Stages: NoiseSuppressor → GainController.
- `NoiseSuppressor` — DeepFilterNet 3 primary, spectral subtraction fallback.
- `GainController` — Smoothed peak AGC with configurable target RMS.
- `SampleRateConverter` — Polyphase resampler.

### `speech/config/`
**Responsibility:** YAML-based typed configuration with deep-merge support.
**Dependencies:** `yaml`.
**Key interfaces:**
- `SpeechConfig` — Root config dataclass containing: `AudioConfig`, `VADConfig`, `WakeWordConfig`, `STTConfig`, `TTSConfig`, `SpeakerConfig`, `MonitoringConfig`, plus `command_map` dict.
- `load_config(user_path)` — Loads defaults from `config/defaults/*.yaml`, deep-merges with user overrides.
- Default YAML files: `speech.yaml`, `models.yaml`, `audio.yaml`, `devices.yaml`, `voice.yaml`, `logging.yaml`.

### `speech/errors/`
**Responsibility:** Exception hierarchy and automatic error recovery.
**Dependencies:** None (stdlib only).
**Key interfaces:**
- `SpeechError` — Base exception. Every exception has a `recoverable: bool` flag.
- Hierarchy: `AudioError` (DeviceNotFoundError, DeviceDisconnectedError, AudioOverflowError), `STTError` (STTTranscriptionError, STTTimeoutError, ModelLoadError), `TTSError`, `VADError`, `WakeWordError`, `ConfigError`.
- `ErrorRecoveryManager` — Registers recovery strategies per exception type. Uses exponential backoff. `.attempt_recovery(error)` → bool.
- `RecoveryAction` — Named recovery callable with `max_retries`.

### `speech/monitoring/`
**Responsibility:** Structured logging and thread-safe metrics collection.
**Dependencies:** `logging` (stdlib).
**Key interfaces:**
- `get_logger(name)` — Returns a configured logger with structured extras.
- `configure_logging(level, log_file)` — Sets up logging handlers.
- `MetricsCollector` (singleton via `get_collector()`) — Thread-safe. `.counter(name)`, `.gauge(name, value)`, `.timer(name)` context manager, `.histogram(name, value)`, `.get_summary()`.

### `speech/vad/`
**Responsibility:** Voice Activity Detection — determines when speech starts/ends.
**Dependencies:** `torch` (for Silero), `numpy`.
**Key interfaces:**
- `BaseVAD` (ABC) — `.process_frame(AudioFrame)` → `VADResult`, `.get_speech_segment()` → `Optional[AudioFrame]`, `.reset()`, `.is_speaking`.
- `VADResult` — `is_speech`, `confidence`, `speech_start`, `speech_end`.
- `SileroVAD` — Production implementation using Silero VAD model. Accumulates frames into speech segments.
- `EnergyVAD` — RMS energy fallback for environments without torch.

### `speech/wakeword/`
**Responsibility:** Always-on wake word detection ("Simon").
**Dependencies:** `openwakeword` (optional).
**Key interfaces:**
- `BaseWakeWordDetector` (ABC) — `.process_frame(AudioFrame)` → `WakeWordResult`, `.reset()`.
- `WakeWordResult` — `detected`, `confidence`, `keyword`.
- `OpenWakeWordDetector` — CNN-based neural detector with cooldown.
- `TextMatchDetector` — Legacy fallback that matches wake word in STT transcript text.

### `speech/stt/`
**Responsibility:** Speech-to-Text engines and confidence enforcement.
**Dependencies:** `faster_whisper`, `ctranslate2`.
**Key interfaces:**
- `BaseSTTEngine` (ABC) — `.transcribe(AudioFrame, DecodingParams)` → `Transcript`, `.transcribe_streaming(...)`, `.load()`, `.unload()`, `.name`, `.is_loaded`.
- `Transcript` — `text`, `raw_text`, `confidence`, `language`, `words: list[WordInfo]`, `avg_log_prob`, `compression_ratio`, `no_speech_prob`, `engine_name`, `latency_ms`.
- `WordInfo` — `word`, `start`, `end`, `confidence`.
- `FasterWhisperEngine` — CTranslate2-accelerated Whisper Large-v3. Lazy-loads model on first transcription.
- `ConfidenceGate` — Per-action confidence thresholds. `.check(Transcript, action)` → `GateResult`. Safety-critical actions require 0.85, navigation 0.80, default 0.60.

### `speech/preprocessing/`
**Responsibility:** Input text normalization, grammar correction, phonetic correction, and intent parsing.
**Dependencies:** `speech.vocabulary`, `speech.models`.
**Key interfaces:**
- `CommandNormalizer` — Filler word removal, verb normalization, punctuation cleanup.
- `GrammarCorrector` — Duplicate word removal, preposition fixes for commands.
- `PhoneticCorrector` — PhoneticDictionary pipeline wrapper.
- `IntentParser` — Multi-strategy parser (regex patterns + fuzzy word-overlap fallback) → `ParsedIntent`.

### `speech/postprocessing/`
**Responsibility:** Filter Whisper hallucinations and artifacts.
**Dependencies:** None (stdlib only).
**Key interfaces:**
- `HallucinationFilter` — `.filter(Transcript)` → `FilterResult`. Checks compression ratio, log probability, no-speech probability, known hallucination patterns (repeated phrases, music notations, timestamps). `FilterResult` has `is_hallucination`, `reason`, `original_transcript`.

### `speech/tts/`
**Responsibility:** Text-to-Speech synthesis with emotional styling, caching, and streaming.
**Dependencies:** `pyttsx3`, optional `TTS` (Coqui), optional `torch`.
**Key interfaces:**
- `BaseTTSEngine` (ABC) — `.synthesize(text, EmotionalStyle)` → `Iterator[AudioFrame]`, `.load()`, `.unload()`, `.name`, `.is_loaded`.
- `NeuralTTSEngine` — Coqui XTTS-v2 wrapper. GPU-accelerated, 24kHz output.
- `Pyttsx3Engine` — OS-native fallback. Saves to temp WAV, reads back as AudioFrame.
- `EmotionalStyle` — Frozen dataclass: `speed_factor`, `pitch_shift_pct`, `volume_pct`, `emphasis_level`, `description`.
- `get_style_for_priority(SpeechPriority)` → `EmotionalStyle` — Maps priority to acoustic properties.
- `TextNormalizer` — Expands abbreviations (Dr. → Doctor, km → kilometers, & → and).
- `PronunciationDictionary` — Regex-based phonetic overrides for domain-specific words.
- `SpeechCache` — In-memory + disk cache keyed by (text, engine, style). LRU eviction at `max_entries`.
- `StreamingSynthesizer` — Orchestrates: normalize → pronunciation → cache check → sentence chunking → engine synthesis → cache store. Yields AudioFrame chunks for streaming playback.

### `speech/queue/`
**Responsibility:** Priority-based speech output queue with preemption.
**Dependencies:** None (stdlib only).
**Key interfaces:**
- `SpeechPriorityQueue` — Thread-safe. `.put(text, priority, on_complete)` → `SpeakItem`. `.get(timeout)` → `SpeakItem`. `.abort_all()`. `.clear_queue()`. Sets `abort_event` when higher-priority item preempts current.
- `SpeakItem` — `priority`, `timestamp`, `_counter`, `text`, `on_complete`, `aborted`. Ordered by (priority, timestamp, counter).

### `speech/manager/`
**Responsibility:** Top-level orchestration facades.
**Dependencies:** All other speech modules.
**Key interfaces:**
- `SpeechManager` — **THE public API**. `.start()`, `.stop()`, `.get_command(timeout)` → `Optional[SpeechCommand]`, `.speak(text, priority)`, `.pause_listening()`, `.resume_listening()`. Constructor takes `SpeechConfig` or loads defaults. Wires all components via constructor DI.
- `ListenPipeline` — Background thread: AudioStream → DSP → VAD → WakeWord → STT → HallucinationFilter → ConfidenceGate → IntentParser → SpeechCommand queue. `.start()`, `.stop()`, `.get_command(timeout)`, `.pause()`, `.resume()`.
- `SpeakPipeline` — Background thread: consumes `SpeechPriorityQueue`, synthesizes via `StreamingSynthesizer`, plays via `sounddevice`. Pauses mic during playback. Supports preemption via `abort_event`.

### `speech/scene/`
**Responsibility:** Acoustic environment classification and adaptive STT decoding parameter selection.
**Key interfaces:** `SceneClassifier`, `AdaptiveDecoder`.

### `speech/context/`
**Responsibility:** Conversational context memory, anaphora resolution, and intelligent re-decoding.
**Key interfaces:** `ContextManager`, `AnaphoraResolver`, `IntelligentRedecoder`.

### `speech/safety/`
**Responsibility:** Pre-execution safety validation, confirmation dialogs, and safety rule enforcement.
**Key interfaces:** `SafetyValidator`, `ConfirmationManager`, `SafetyRule`.

### `speech/vocabulary/`
**Responsibility:** Domain vocabulary aggregation, OCR session vocab, and phonetic corrections.
**Key interfaces:** `VocabularyManager`, `LocationVocab`, `UserVocab`, `SessionVocab`, `PhoneticDictionary`.

### `speech/adaptation/`
**Responsibility:** Persistent user profiles, frequency-weighted command history, correction learner, and accent adaptation.
**Key interfaces:** `UserProfileManager`, `CommandHistory`, `CorrectionLearner`, `AccentAdapter`.

### `speech/speaker/`
**Responsibility:** Speaker verification and multi-sample voiceprint enrollment.
**Key interfaces:** `EcapaVerifier`, `VoiceprintEnrollment`, `BaseSpeakerVerifier`.

### `speech/monitoring/`
**Responsibility:** Structured logging, thread-safe metrics collection, component health monitoring, and TUI dashboard.
**Key interfaces:** `get_logger()`, `get_collector()`, `HealthMonitor`, `SpeechDashboard`.

### `speech/tests/`
**Responsibility:** Automated unit, integration, and stress tests.
**Dependencies:** `pytest`.
**Structure:** `conftest.py` (shared fixtures, synthetic audio factories), `unit/` (16 test files, 258 unit tests), `integration/` (2 test files, 16 integration tests), `stress/` (empty). Total: 274 tests passing.

### `speech/benchmarks/`
**Responsibility:** Performance benchmarking (WER, latency, resource usage).
**Status:** Empty `__init__.py` only (future enhancement).

---

## File Reading Order

When starting a new session, read files in this order:

1. **`CLAUDE.md`** (this file) — Understand project state, architecture, conventions, and what NOT to touch.

2. **`implementation_plan.md`** (in artifacts directory) — Full design document with architecture diagrams, design decisions, and module specifications for all phases. Read to understand the planned architecture beyond what's implemented.

3. **`task.md`** (in artifacts directory) — Checklist of completed and remaining work items. Check this to know exactly what's done.

4. **`speech/models/*`** — Data models are the shared vocabulary. Every other module depends on AudioFrame, SpeechPriority, SpeechCommand. Read these first to understand the data flow.

5. **`speech/errors/*`** — Exception hierarchy determines recovery strategies. Understanding which errors are recoverable shapes how you write new code.

6. **`speech/monitoring/*`** — Logger and metrics patterns. Every module you write must use `get_logger()` and `get_collector()`. Read to match the established logging style.

7. **`speech/config/*`** — Configuration dataclasses and YAML loader. New features must add config fields here. Understand the deep-merge loading pattern.

8. **`speech/audio/*`** — Audio I/O layer. Understand AudioStream's callback model, DSP chain-of-responsibility, and how device hot-swap works.

9. **`speech/vad/*`** — VAD interface and implementations. Understand `process_frame()` → `VADResult` → `get_speech_segment()` lifecycle.

10. **`speech/wakeword/*`** — Wake word detection interface. Simple but critical for understanding the listen pipeline flow.

11. **`speech/stt/*`** — STT engine interface, Whisper implementation, and confidence gating. Core recognition logic.

12. **`speech/postprocessing/*`** — Hallucination filter. Understand what it catches and why.

13. **`speech/tts/*`** — TTS engines, emotional profiles, caching, streaming. Understand the synthesis chain.

14. **`speech/queue/*`** — Priority queue with preemption semantics. Understanding abort_event is critical.

15. **`speech/manager/*`** — Read last. SpeechManager wires everything together. ListenPipeline and SpeakPipeline are the runtime orchestrators.

16. **`speech/tests/*`** — Read `conftest.py` to understand test fixtures and mocking patterns before writing new tests.

---

## Public APIs

These interfaces are **contracts**. Future development must preserve their signatures.

### Core Data Models
- `AudioFrame(data, sample_rate, channels, dtype, timestamp, device_id, sequence_id)`
  - Properties: `.duration_s`, `.num_samples`, `.rms()`
  - Methods: `.to_float32()`, `.to_int16()`, `.resample(target_rate)`, `.concatenate(frames)` (classmethod)
- `SpeechPriority` — IntEnum: EMERGENCY(1), OBSTACLE(2), NAVIGATION(3), OCR(4), FACE(5), STATUS(6), NOTIFICATION(7), INFO(8), CONFIRMATION(9), DEBUG(10)
  - Properties: `.should_interrupt`, `.is_safety_critical`, `.label`
  - Classmethod: `.from_legacy(int)`
- `SpeechCommand(action, args, confidence, raw_transcript, source, status, ...)`
  - Supports `cmd["action"]`, `cmd.get("key", default)` for backward compat with `main.py`
- `SceneType` — Enum: QUIET_ROOM, OFFICE, STREET, VEHICLE, CROWD, OUTDOOR_PARK, INDOOR_MALL, TRAIN_STATION
- `DecodingParams(beam_size, temperature, patience, best_of, language, initial_prompt, ...)`

### Abstract Base Classes
- `BaseSTTEngine` — `.transcribe()`, `.transcribe_streaming()`, `.load()`, `.unload()`, `.name`, `.is_loaded`
- `BaseTTSEngine` — `.synthesize(text, style)` → `Iterator[AudioFrame]`, `.load()`, `.unload()`, `.name`, `.is_loaded`
- `BaseVAD` — `.process_frame()` → `VADResult`, `.get_speech_segment()`, `.reset()`, `.is_speaking`
- `BaseWakeWordDetector` — `.process_frame()` → `WakeWordResult`, `.reset()`

### Result Types
- `VADResult(is_speech, confidence, speech_start, speech_end)`
- `WakeWordResult(detected, confidence, keyword)`
- `Transcript(text, raw_text, confidence, language, words, avg_log_prob, compression_ratio, no_speech_prob, engine_name, latency_ms)`
- `WordInfo(word, start, end, confidence)`
- `GateResult(passed, action, threshold_used, confidence, reason)`
- `FilterResult(is_hallucination, reason, original_transcript)`
- `EmotionalStyle(speed_factor, pitch_shift_pct, volume_pct, emphasis_level, description)`

### Manager Facade
- `SpeechManager(config, user_config_path)`
  - `.start()`, `.stop()`
  - `.get_command(timeout)` → `Optional[SpeechCommand]`
  - `.speak(text, priority)`
  - `.pause_listening()`, `.resume_listening()`
  - `.config`, `.is_running`

### Pipeline Internals
- `ListenPipeline` — `.start()`, `.stop()`, `.get_command(timeout)`, `.pause()`, `.resume()`
- `SpeakPipeline` — `.start()`, `.stop()`
- `SpeechPriorityQueue` — `.put(text, priority, on_complete)`, `.get(timeout)`, `.abort_all()`, `.clear_queue()`, `.abort_event`
- `StreamingSynthesizer` — `.synthesize_stream(text, style)` → `Iterator[AudioFrame]`

### Infrastructure
- `ErrorRecoveryManager` — `.register(exception_type, RecoveryAction)`, `.attempt_recovery(error)`
- `MetricsCollector` — `.counter()`, `.gauge()`, `.timer()`, `.histogram()`, `.get_summary()`
- `get_logger(name)`, `configure_logging(level, log_file)`
- `load_config(user_path)` → `SpeechConfig`

---

## Architecture Rules

### SOLID Principles
- **SRP**: Each file has one responsibility. Confidence gating is separate from hallucination filtering is separate from safety validation.
- **OCP**: New STT engines extend `BaseSTTEngine`; new TTS engines extend `BaseTTSEngine`. No modification to existing code.
- **LSP**: All VAD implementations are interchangeable through `BaseVAD`.
- **ISP**: `BaseSTTEngine` has both `transcribe()` (required) and `transcribe_streaming()` (optional with default impl).
- **DIP**: `SpeechManager` depends on abstractions (`BaseVAD`, `BaseSTTEngine`), not concrete implementations.

### Dependency Injection
- All components receive dependencies via constructor parameters
- `SpeechManager.start()` is the composition root that wires everything
- No service locator, no global registries (except `MetricsCollector` singleton, by design)

### Structured Logging
- Use `get_logger("module.submodule")` for every module
- Log at appropriate levels: ERROR for failures, WARNING for fallbacks, INFO for lifecycle, DEBUG for frame-level
- Include structured extras in `extra={}` dict (never string interpolation in log messages for structured data)

### Metrics Collection
- Use `metrics.counter("namespace.event")` for counts
- Use `metrics.timer("namespace.operation")` as context manager for latency
- Namespace convention: `audio.*`, `vad.*`, `stt.*`, `tts.*`, `pipeline.*`

### Thread Safety
- `AudioStream` callback runs on PortAudio's real-time thread — no allocations, no blocking, no logging
- `ListenPipeline` and `SpeakPipeline` run in dedicated daemon threads
- All shared state protected by `threading.Lock`
- Inter-thread communication via `queue.Queue` (never shared mutable state)

### Error Handling
- All speech exceptions inherit from `SpeechError` with `recoverable` flag
- Catch specific exceptions, not bare `except Exception`
- Log errors before recovery attempt
- Never silence exceptions without logging

### Recovery Strategy
- `ErrorRecoveryManager` maps exception types to `RecoveryAction` callables
- Exponential backoff between retries
- Max retry count per strategy
- Auto-disable component after max retries exhausted

### Configuration Management
- All parameters in `SpeechConfig` dataclasses with validation in `__post_init__`
- Defaults in `config/defaults/*.yaml`
- User overrides via deep-merge from user YAML path
- Never hardcode magic numbers — always reference config

### Testing Requirements
- Every module gets unit tests in `tests/unit/`
- Use `pytest.importorskip()` for optional heavy dependencies (torch, sounddevice)
- Mock external dependencies (sounddevice, torch models) in unit tests
- Test fixtures in `conftest.py` — reuse `make_audio_frame()`, `make_silence()`, `make_sine_wave()`
- Target: 90%+ code coverage for safety-critical paths

### Type Hints
- `from __future__ import annotations` in every file
- Full type hints on all public methods
- `Optional[]` for nullable returns
- `Iterator[]` for generators

### Documentation Style
- Module-level docstring explaining purpose and design decisions
- Class-level docstring with Attributes section
- Method docstrings with Args, Returns, Raises sections
- Design decisions documented as comments starting with "Design decision:"

### Performance Requirements
- STT latency: < 500ms for 3-second utterance on GPU
- TTS first-chunk latency: < 200ms (streaming)
- VAD processing: < 1ms per frame
- DSP pipeline: < 5ms per frame
- Wake word detection: < 3ms per frame

### Backward Compatibility
- `SpeechCommand` supports `cmd["action"]` dict-style access (for `main.py`)
- `SpeechPriority.from_legacy(int)` bridges old integer priorities
- Legacy files (`speech_input.py`, `speech_output.py`) kept in root for reference

---

## Design Decisions

**DO NOT change these without explicit user approval:**

1. **faster-whisper over OpenAI Whisper** — 4× faster, same accuracy, lower VRAM via CTranslate2
2. **Silero VAD over webrtcvad** — State-of-the-art accuracy, better at detecting speech boundaries
3. **OpenWakeWord over text matching** — Dedicated CNN runs always-on without requiring full STT
4. **Constructor DI over service locator** — Explicit deps, testable, no hidden coupling
5. **Chain-of-responsibility DSP** — Stages can be independently enabled/disabled via config
6. **Background thread pipelines** — Decouples audio capture from processing from command handling
7. **Priority queue with preemption** — Emergency speech MUST interrupt lower-priority output
8. **Emotional TTS profiles** — Acoustic urgency cues are critical for visually impaired users
9. **Sentence-level streaming TTS** — Reduces perceived latency without destroying prosody
10. **YAML config with dataclass validation** — Type-safe, human-readable, overridable, no Pydantic hard dep
11. **AudioFrame as universal currency** — Single type flows through entire pipeline, prevents sample-rate mismatches
12. **Recoverable flag on exceptions** — Enables automatic recovery for transient errors without human intervention
13. **Hallucination filter before confidence gate** — Filter known bad patterns before threshold checking

---

## Testing Status

**Full Suite Results:**
- **274 passed, 2 skipped, 0 failures** (across 18 test files)
- Skipped: 1 in `test_device_manager.py` (requires `sounddevice`), 1 in `test_vad.py` (requires `torch`)
- Tests use `pytest.importorskip()` for graceful degradation

**Test Breakdown by Phase:**
- **Phase 1 (Core & STT):** 85 unit tests (`test_device_manager`, `test_dsp_pipeline`, `test_vad`, `test_confidence_gate`, `test_hallucination_filter`)
- **Phase 3 (Multi-Engine & Scene):** 50 unit tests (`test_scene_classifier`, `test_adaptive_decoder`, `test_intelligent_redecoder`, `test_engine_orchestrator`)
- **Phase 4 (Context, Safety, Vocab, Preprocessing):** 90 unit tests (`test_context_manager`, `test_anaphora_resolver`, `test_safety_validator`, `test_vocabulary_manager`, `test_phonetic_dictionary`)
- **Phase 5 (Adaptation, Speaker, Dashboard, Integration):** 33 unit tests (`test_personal_adapter`) + 16 integration tests (`test_listen_pipeline`, `test_speech_manager`)

**Development environment:**
- Windows 10/11
- Python 3.10+
- Virtual environment at `speech/.venv` or project root `.venv`
- `sounddevice` and `torch` may not be installed on dev machine — code must handle ImportError gracefully
- GPU (CUDA) optional — code auto-detects and falls back to CPU

**Required dependencies (core):** `numpy`, `pyyaml`, `pyttsx3`
**Optional dependencies:** `faster-whisper`, `torch`, `sounddevice`, `scipy`, `openwakeword`, `deepfilternet`, `TTS` (Coqui)

---

## Completed Files

### `speech/models/` (5 files)
- `audio_frame.py` — Universal audio data container
- `speech_priority.py` — Priority enum with safety classification
- `speech_command.py` — Parsed command with backward-compat dict access
- `scene_type.py` — Acoustic environment enum
- `decoding_params.py` — Adaptive STT decoding parameters

### `speech/audio/` (6 files)
- `device_manager.py` — Device enumeration, WDM-KS ranking, hot-swap monitoring
- `audio_stream.py` — PortAudio callback stream with ring buffer
- `sample_rate_converter.py` — Polyphase resampler
- `noise_suppressor.py` — DeepFilterNet 3 + spectral fallback
- `gain_controller.py` — Smoothed peak AGC
- `dsp_pipeline.py` — Chain-of-responsibility DSP orchestrator

### `speech/config/` (9 files)
- `speech_config.py` — 8 typed config dataclasses (added SceneConfig)
- `loader.py` — YAML deep-merge loader
- `defaults/speech.yaml` — Master config + command vocabulary
- `defaults/models.yaml` — STT/speaker model params
- `defaults/audio.yaml` — Audio pipeline params
- `defaults/devices.yaml` — Device preferences
- `defaults/voice.yaml` — TTS voice params
- `defaults/logging.yaml` — Monitoring params

### `speech/errors/` (2 files)
- `exceptions.py` — Full exception hierarchy with recoverable flags
- `recovery.py` — ErrorRecoveryManager with exponential backoff

### `speech/monitoring/` (2 files)
- `logger.py` — Structured logging setup
- `metrics.py` — Thread-safe MetricsCollector singleton

### `speech/vad/` (3 files)
- `base.py` — BaseVAD + VADResult
- `silero_vad.py` — Production Silero VAD
- `energy_vad.py` — RMS energy fallback

### `speech/wakeword/` (3 files)
- `base.py` — BaseWakeWordDetector + WakeWordResult
- `openwakeword_detector.py` — Neural wake word detection
- `text_match_detector.py` — Text-matching fallback

### `speech/stt/` (8 files)
- `base.py` — BaseSTTEngine ABC
- `transcript.py` — Transcript + WordInfo models
- `faster_whisper_engine.py` — faster-whisper CTranslate2 engine
- `confidence_gate.py` — Per-action confidence thresholds
- `engine_registry.py` — Plug-in engine registration, health tracking, priority ordering
- `intelligent_redecoder.py` — Multi-temperature redecoding with consensus selection
- `adaptive_decoder.py` — Scene-driven decoding parameter mapping
- `engine_orchestrator.py` — Multi-engine STT orchestrator with fallback chain

### `speech/postprocessing/` (1 file)
- `hallucination_filter.py` — Whisper hallucination detection

### `speech/tts/` (8 files)
- `base.py` — BaseTTSEngine ABC
- `emotional_profile.py` — SpeechPriority → EmotionalStyle mapping
- `neural_tts_engine.py` — Coqui XTTS-v2 engine
- `pyttsx3_engine.py` — pyttsx3 fallback engine
- `text_normalizer.py` — Abbreviation expansion
- `pronunciation_dict.py` — Phonetic overrides
- `speech_cache.py` — In-memory + disk audio cache
- `streaming_synthesizer.py` — Chunk-and-stream pipeline

### `speech/queue/` (1 file)
- `priority_queue.py` — Priority queue with preemption

### `speech/manager/` (3 files)
- `listen_pipeline.py` — Audio → command pipeline thread
- `speak_pipeline.py` — Queue → synthesis → playback thread
- `speech_manager.py` — Unified facade wiring all components

### `speech/scene/` (4 files)
- `base.py` — BaseSceneClassifier ABC
- `scene_profile.py` — SceneProfile dataclass (scene type, SNR, reverb, confidence)
- `scene_classifier.py` — Heuristic classifier with EMA smoothing and majority voting
- `adaptive_controller.py` — Maps SceneProfile → per-component parameter adaptations

### `speech/context/` (4 files)
- `__init__.py`
- `context_store.py` — TTL-based slot storage with eviction and lazy expiry
- `anaphora_resolver.py` — Resolves deictic/anaphoric refs (there, that, again, him)
- `context_manager.py` — Orchestrates context lifecycle and resolution

### `speech/safety/` (4 files)
- `__init__.py`
- `safety_rules.py` — SafetyLevel enum + per-action SafetyRule definitions
- `confirmation_manager.py` — Voice confirmation dialog state machine (IDLE→WAITING→CONFIRMED/DENIED)
- `safety_validator.py` — Pre-execution safety gate with argument pattern matching

### `speech/vocabulary/` (6 files)
- `__init__.py`
- `location_vocab.py` — Location-specific terms (MRU ordering, capped)
- `user_vocab.py` — User-defined vocabulary (JSON file persistence)
- `session_vocab.py` — Session-scoped dynamic terms (OCR text, labels)
- `phonetic_dictionary.py` — Multi-pronunciation lookup + word-by-word correction
- `vocabulary_manager.py` — Aggregates all vocabs into ~200-token STT prompt

### `speech/preprocessing/` (5 files)
- `__init__.py`
- `command_normalizer.py` — Filler removal, verb normalization, punctuation cleanup
- `grammar_corrector.py` — Duplicate word removal, preposition fixes
- `phonetic_corrector.py` — PhoneticDictionary pipeline wrapper
- `intent_parser.py` — Multi-strategy intent parser (pattern + fuzzy matching)

### `speech/adaptation/` (5 files)
- `__init__.py`
- `user_profile.py` — UserProfile dataclass + UserProfileManager (JSON persistence)
- `command_history.py` — Frequency-weighted command history (24h recency half-life)
- `correction_learner.py` — Learn from user corrections, extends PhoneticDictionary + profile
- `accent_adapter.py` — Accent-specific Whisper prompt tuning (5 built-in accents)

### `speech/speaker/` (4 files)
- `__init__.py`
- `base.py` — BaseSpeakerVerifier ABC (enroll/verify/extract_embedding)
- `ecapa_verifier.py` — ECAPA-TDNN verifier (lazy SpeechBrain, cosine similarity)
- `enrollment.py` — Multi-sample enrollment workflow with voiceprint .npy persistence

### `speech/monitoring/` (4 files)
- `__init__.py`
- `logger.py` — Structured logging with correlation IDs
- `metrics.py` — Thread-safe counters, gauges, histograms
- `health_monitor.py` — Component health registration and throttled checking
- `dashboard.py` — TUI status dashboard with custom panels

### `speech/audio/` (4 files)
- `__init__.py`
- `audio_stream.py` — Thread-safe ring buffer for audio
- `dsp_pipeline.py` — Chain-of-responsibility DSP processing
- `echo_canceller.py` — Energy-based echo cancellation

### `speech/tests/` (18 files)
- `conftest.py` — Shared fixtures and audio factories
- `unit/test_device_manager.py` — 14 tests
- `unit/test_dsp_pipeline.py` — 16 tests
- `unit/test_vad.py` — 12 tests
- `unit/test_confidence_gate.py` — 14 tests
- `unit/test_hallucination_filter.py` — 15 tests
- `unit/test_scene_classifier.py` — 12 tests
- `unit/test_adaptive_decoder.py` — 12 tests
- `unit/test_intelligent_redecoder.py` — 8 tests
- `unit/test_engine_orchestrator.py` — 18 tests
- `unit/test_context_manager.py` — 18 tests
- `unit/test_anaphora_resolver.py` — 13 tests
- `unit/test_safety_validator.py` — 24 tests
- `unit/test_vocabulary_manager.py` — 19 tests
- `unit/test_phonetic_dictionary.py` — 16 tests
- `unit/test_personal_adapter.py` — 33 tests
- `integration/test_listen_pipeline.py` — 9 tests
- `integration/test_speech_manager.py` — 7 tests

**Total: 274 tests passing, 0 failures**

### `speech/_legacy/` (2 files)
- `speech_input.py` — Original monolithic STT (archived)
- `speech_output.py` — Original monolithic TTS (archived)

---

## Speech Subsystem Complete

All 5 speech implementation phases are complete. The speech subsystem is fully built and tested.
Remaining speech work is hardware integration testing on the physical SIMON device.

### Speech Future Enhancements (not planned)
- Stress tests (noise robustness, queue overflow, recovery under load)
- WER benchmarking against standard datasets
- Real-time CPU/GPU profiling

---

## System Architecture Summary

The full system architecture is documented in `implementation_plan.md`. Key architectural decisions:

### New Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Event Bus** | `core/events/event_bus.py` | Pub/sub inter-subsystem communication with priority and topic filtering |
| **SystemStateMachine** | `core/state/state_machine.py` | 11 states (STARTING→STOPPED) with validated transitions |
| **CapabilityRegistry** | `core/capabilities/registry.py` | Single source of truth for runtime feature availability |
| **AppController** | `core/app_controller.py` | Top-level orchestrator; main.py delegates entirely to this class |
| **Perception Fusion** | `vision/pipeline/perception_fusion.py` | Merges YOLO+Face+OCR+Scene into unified WorldModel |
| **Logic Controller** | `core/logic/logic_controller.py` | Event-driven decision engine (perception→decision→action) |
| **Safety Engine** | `core/safety/safety_engine.py` | 4-level hazard classification with emergency TTS override |
| **Task Planner** | `core/planner/task_planner.py` | Decomposes voice commands into executable sub-tasks |
| **Action Executor** | `core/actions/action_executor.py` | Maps abstract actions to subsystem handlers |
| **Plugin System** | `core/plugins/plugin_manager.py` | Plugin discovery, lifecycle, event-based hooks |

### Architecture Principles
- **Event-driven:** Subsystems communicate via Event Bus, not direct imports
- **Safety-first:** Hazard events bypass queues for immediate TTS
- **Offline-first:** Every critical path works without internet
- **DI everywhere:** Constructor injection, no service locators
- **Minimal main.py:** Bootstrap only (~15 lines); orchestration in AppController
- **9 dedicated threads:** Camera, Vision, OCR, Event Bus, Navigation, Speech Listen/Speak (frozen), Watchdog, Main

### Technology Decision Records
10 TDRs documented in `implementation_plan.md` Section 17, covering:
Event Bus, DI, CapabilityRegistry, StateMachine, Perception Fusion, Plugin System, Threading Model, Depth Estimation, Minimal main.py, Resource Management

---

## Files That Should Rarely Be Modified

These files are stable and tested. Modify only for bug fixes or architecture-level changes:

- `speech/models/audio_frame.py`
- `speech/models/speech_priority.py`
- `speech/models/speech_command.py`
- `speech/errors/exceptions.py`
- `speech/errors/recovery.py`
- `speech/monitoring/logger.py`
- `speech/monitoring/metrics.py`
- `speech/config/loader.py`
- `speech/vad/base.py`
- `speech/wakeword/base.py`
- `speech/stt/base.py`
- `speech/stt/transcript.py`
- `speech/tts/base.py`
- `speech/audio/audio_stream.py`
- `speech/audio/dsp_pipeline.py`
- `speech/tests/conftest.py`

---

## Instructions For Future Claude

1. **Read before coding.** Read this file, then `task.md`, then `implementation_plan.md`. Only then open source files.
2. **Never regenerate completed modules.** All files listed under "Completed Files" are done. Do not rewrite them.
3. **Continue from existing implementations.** New code extends existing patterns. Match the style of adjacent modules.
4. **Preserve all public APIs.** Every interface listed under "Public APIs" is a contract. Do not change signatures.
5. **Preserve the architecture.** Constructor DI, ABC base classes, chain-of-responsibility DSP, facade pattern for SpeechManager.
6. **Preserve coding conventions.** `from __future__ import annotations`, `get_logger()`, `get_collector()`, structured logging, dataclasses over plain dicts.
7. **Complete one phase at a time.** Do not start Phase N+1 until Phase N is done with tests.
8. **Update CLAUDE.md** after completing each phase (add files to "Completed Files", update "Current Progress" table).
9. **Update task.md** after every completed milestone (mark items `[x]`).
10. **Never rewrite working code** without a bug report or explicit architectural change request.
11. **Write tests** for every new module in `tests/unit/`. Follow the existing fixture patterns in `conftest.py`.
12. **Handle ImportError gracefully** for optional heavy dependencies (torch, sounddevice, TTS, deepfilternet).
13. **Add config fields** to `SpeechConfig` dataclasses when adding new configurable features.
14. **Log everything** at appropriate levels. Use metrics counters for events and timers for latency.

---

## Resume Prompt

Copy and paste this prompt to start a new Claude session on this project:

```
You are continuing development on SIMON — an AI-powered assistive system for visually impaired users.

1. Read `CLAUDE.md` in the project root first. It contains the full project knowledge base.
2. Read `task.md` to see what's completed and what remains.
3. Read `implementation_plan.md` for the complete architecture document.
4. The Speech subsystem is COMPLETE and FROZEN. Do NOT modify any files in speech/.
5. Check "Current Progress" in CLAUDE.md to find the current system phase.
6. Resume from the current system phase. Do NOT regenerate completed modules.
7. Preserve all public APIs, architecture patterns, coding conventions, and dependency injection.
8. Follow the architecture document exactly. Do NOT redesign subsystems.
9. Complete the current phase fully, including unit tests.
10. After completing a phase, update both `CLAUDE.md` and `task.md`.
11. Use CapabilityRegistry for feature checks, not scattered try/except.
12. Use StateMachine for lifecycle state, not scattered booleans.
13. Use Event Bus for inter-subsystem communication, not direct imports.
14. Keep main.py as minimal bootstrap. Orchestration goes in AppController.
15. Maintain production-quality engineering standards throughout.
```
