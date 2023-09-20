# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import torchaudio
import random
import subprocess
from torch import Tensor
from flask import Flask, render_template, request, flash
from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import (
    SpeechTransformer,
    Jasper,
    DeepSpeech2,
    ListenAttendSpell,
    Conformer,
)

app = Flask(import_name=__name__)

if torch.cuda.is_available():
    ondevice = 'cuda'
else:
    ondevice = 'cpu'

path_dir = '*********Input audio path for inference*********'

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part in the form.'
    file = request.files['file']

    if file.filename == '':
        return 'No selected file.'
    file.save(os.path.join("*********Input audio path for inference*********", file.filename))

    file_name = path_dir+file.filename
    
    app.config["AUDIO_PATH"] = file_name
    return render_template('infer.html')


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    if torch.cuda.is_available():
        return torch.cuda.FloatTensor(feature).transpose(0, 1) # --device='cuda' inference
    else:
        return torch.FloatTensor(feature).transpose(0, 1) # --device='cpu' inference



@app.route('/infer', methods=['POST'])
def infer():
    audio_file_path = app.config["AUDIO_PATH"]
    parser = argparse.ArgumentParser(description='KoSpeech')
    parser.add_argument('--model_path', type=str, required=False, default='*********Input Model Path for inference*********')
    parser.add_argument('--device', type=str, required=False, default=ondevice)
    opt = parser.parse_args()


    feature = parse_audio(audio_file_path, del_silence=True)
    input_length = torch.LongTensor([len(feature)])
    vocab = KsponSpeechVocabulary('*********Input Vocab Path for inference*********')

    model = torch.load(opt.model_path, map_location=lambda storage, loc: storage).to(opt.device)
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    if isinstance(model, ListenAttendSpell):
        model.encoder.device = opt.device
        model.decoder.device = opt.device

        y_hats = model.recognize(feature.unsqueeze(0), input_length)
    elif isinstance(model, DeepSpeech2):
        model.device = opt.device
        y_hats = model.recognize(feature.unsqueeze(0), input_length)
    elif isinstance(model, SpeechTransformer) or isinstance(model, Jasper) or isinstance(model, Conformer):
        y_hats = model.recognize(feature.unsqueeze(0), input_length)

    sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
    result_sentence = ','.join(sentence)
    
    for file in os.scandir(path_dir):
        os.remove(file.path)

    return result_sentence

if __name__ == '__main__':
    app.run(debug=True, port=8000)
