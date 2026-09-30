from recorder import AudioRecorder 
import whisper
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import resample_poly
import warnings
import keyboard

warnings.filterwarnings("ignore")

class SpeechToText:
    def __init__(self):
        self.model = whisper.load_model("small.en")

    def transcription(self,audio):
        audio_16k = resample_poly(audio, 1, 3).astype(np.float32)

        result = self.model.transcribe(audio_16k, language="en",temperature=0)
        print(result["text"])


recorder = AudioRecorder()
print("Hold P to record. Release P to stop.")

keyboard.wait('p')
recorder.start()
print("start recording...")

keyboard.wait('p')
recorder.stop()
print("stop recording.")
record = recorder.merged_chunks
write("test.wav", 48000, record)
stt = SpeechToText()
stt.transcription(record)



