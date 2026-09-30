#  .venv\Scripts\Activate.ps1
import sounddevice as sd
import numpy as np


class AudioRecorder:
    
    def __init__(self):
        self.recording = False
        self.rate = 48000
        self.nb_channels = 1
        self.chunks = []

        self.record = sd.InputStream(
          samplerate=self.rate,
          channels=self.nb_channels,
          device=1,
          callback=self.callback,                                                
        )
         
        
    def callback(self, indata, frames, time, status):
        self.chunks.append(indata.copy())
    
    def start(self):
        self.chunks =[]
        self.record.start()
        self.recording = True 

    def stop(self):
        self.record.stop()
        self.recording = False 
        self.merged_chunks_old = np.concatenate(self.chunks,axis=0)
        self.merged_chunks = np.squeeze(self.merged_chunks_old)


