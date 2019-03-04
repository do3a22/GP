import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc

def preprocess(wav_file, n_input = 26, n_padding = 9):
    sampleRate, audioData = wav.read(wav_file)
    features = mfcc(audioData, samplerate=sampleRate, winlen=0.032, winstep=0.02)
    padding = np.zeros((n_padding, n_input), dtype=features.dtype)
    features = np.concatenate((padding, features, padding))
    return features
