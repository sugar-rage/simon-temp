import sounddevice as sd
import soundfile as sf
import whisper

print("Recording for 5 seconds...")

audio = sd.rec(
    int(5 * 16000),
    samplerate=16000,
    channels=1,
    dtype="float32"
)

sd.wait()

sf.write("test.wav", audio, 16000)

print("Transcribing...")

model = whisper.load_model("base")
result = model.transcribe("test.wav")

print(result["text"])