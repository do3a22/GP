import os
import sys
import wave
from deepspeech import Model
import numpy as np
import subprocess
import shlex
try:
    from shhlex import quote
except ImportError:
    from pipes import quote

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_ALPHA = 0.75

# The beta hyperparameter of the CTC decoder. Word insertion bonus.
LM_BETA = 1.85

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9


def convert_samplerate(audio_path):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate 16000 --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path))
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use 16kHz files or install it: {}'.format(e.strerror))

    return 16000, np.frombuffer(output, np.int16)

def load_model(model,  alphabet, LM = None, LMtrie = None):
    print('loading model..')
    modelInstance = Model(model, N_FEATURES, N_CONTEXT, alphabet, BEAM_WIDTH)
    if ((LM != None) and (LMtrie != None)):
        modelInstance.enableDecoderWithLM(alphabet, LM, LMtrie, LM_ALPHA, LM_BETA)
    print('Model loaded !')
    return modelInstance


def predict(input_wav_file, ds):

    fin = wave.open(input_wav_file, 'rb')
    fs = fin.getframerate()
    if fs != 16000:
        print('Warning: original sample rate ({}) is different than 16kHz. Resampling might produce erratic speech recognition.'.format(fs), file=sys.stderr)
        fs, audio = convert_samplerate(input_wav_file)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    audio_length = fin.getnframes() * (1/16000)
    fin.close()

    print('Running inference..')
    output = ds.stt(audio, fs)
    print('Output text: ' + output)

    return output



wavFile = 'audio/This_is_our_graduation_project.wav'
modelFile = 'models/output_graph.pbmm'
alphabetFile = 'models/alphabet.txt'
lmFile = 'models/lm.binary'
trieFile = 'models/trie'

ds = load_model(modelFile, alphabetFile, lmFile, trieFile)
recognized = predict(wavFile, ds)
