import os

import torch
from torch import nn

import numpy as np

from collections import OrderedDict

from TTS.models.tacotron import Tacotron
from TTS.utils.audio import AudioProcessor
from TTS.utils.text import text_to_sequence
from TTS.utils.generic_utils import load_config
from TTS.layers import *


def load_model(model_path = 'model/best_model.pth.tar',
        config_path = 'model/config.json'):
    CONFIG = load_config(config_path)
    model = Tacotron(CONFIG.embedding_size, CONFIG.num_freq, CONFIG.num_mels, CONFIG.r)
    cp = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(cp['model'])
    model.eval()
    model.decoder.max_decoder_steps = 250
    ap = AudioProcessor(CONFIG.sample_rate, CONFIG.num_mels, CONFIG.min_level_db,
                    CONFIG.frame_shift_ms, CONFIG.frame_length_ms,
                    CONFIG.ref_level_db, CONFIG.num_freq, CONFIG.power, CONFIG.preemphasis,
                    griffin_lim_iters=50)
    text_cleaner = [CONFIG.text_cleaner]
    return model, ap, text_cleaner

def tts(text, textCleaner, model, ap):
    seq = np.array(text_to_sequence(text, text_cleaner))
    chars_var = torch.from_numpy(seq).unsqueeze(0)
    linear_out= model.forward(chars_var.long())
    linear_out = linear_out[0].data.cpu().numpy()
    waveform = ap.inv_spectrogram(linear_out.T)
    waveform = waveform[:ap.find_endpoint(waveform)]
    out_path = 'samples/'
    os.makedirs(out_path, exist_ok=True)
    file_name = text.replace(" ", "_").replace(".","") + ".wav"
    out_path = os.path.join(out_path, file_name)
    ap.save_wav(waveform, out_path)

    return waveform

model, ap, text_cleaner = load_model()

tts("This is our graduation project .", text_cleaner, model, ap)
