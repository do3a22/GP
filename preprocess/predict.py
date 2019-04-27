from deepspeech import Model
import scipy.io.wavfile as wav
import sys

def predict(input_wav_file):
    modelInstance = Model('models/output_graph.pbmm', 26, 9, 'models/alphabet.txt', 500)
    sampleRate, audioData = wav.read(input_wav_file)
    output = modelInstance.stt(audioData, sampleRate)
    return output


recognized = predict('audio/2830-3980-0043.wav')
with open('output/text.txt', 'a') as f:
    f.write(recognized)
