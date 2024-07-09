import pickle
import glob, os
import pdb
import numpy as np
import torch

from whisperx.vad import load_vad_model
from whisperx.asr import SAMPLE_RATE
import whisperx


use_gpu = False
device = "cuda:0" if use_gpu else "cpu"
vad = load_vad_model(device, vad_onset=0.500, vad_offset=0.363, use_auth_token=None, model_fp=None)
in_dir = 'dependencies/annotation_gauntlet/session_0'

output_dir = 'test/whisperx_vad_fun/audio/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# save out .wav files to .npz files since .wav won't load in Jupyter Lab
for wav in glob.glob(in_dir + '/*.wav'):
    print(wav)
    audio = whisperx.load_audio(wav)
    with open(output_dir + os.path.split(wav)[-1].split('.')[0] + '.npy', 'wb') as f:
        np.save(f, audio)
    
    pred = vad({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})

print(pred)
# pdb.set_trace()
