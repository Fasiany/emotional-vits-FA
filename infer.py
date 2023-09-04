import matplotlib.pyplot as plt
import IPython.display as ipd
import playsound

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
import numpy as np


def get_text(text, hps, symbols):
    text_norm = text_to_sequence(text, symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm
hps = utils.get_hparams_from_file("./configs/modified_finetune_speaker.json")
symbols_ = hps['symbols']
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
_ = net_g.eval()
global_steps = 10000
speaker = ""
_ = utils.load_checkpoint(f"logs/{speaker}/G_{global_steps}.pth", net_g, None)
random_emotion_root = "./ATRI_VD_WAV"
import random
def tts(txt, emotion, symbol):
    """emotion为参考情感音频路径 或random_sample（随机抽取）"""
    global emo
    stn_tst = get_text(txt, hps, symbol)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        sid = torch.LongTensor([0])
        if os.path.exists(f"{emotion}.emo.npy"):
            emo = torch.FloatTensor(np.load(f"{emotion}.emo.npy")).unsqueeze(0)
        elif emotion == "random_sample":
            while True:
                rand_wav = random.sample(os.listdir(random_emotion_root), 1)[0]
                if rand_wav.endswith('wav') and os.path.exists(f"{random_emotion_root}/{rand_wav}.emo.npy"):
                    break
            emo = torch.FloatTensor(np.load(f"{random_emotion_root}/{rand_wav}.emo.npy")).unsqueeze(0)
            print(f"{random_emotion_root}/{rand_wav}")
        elif emotion.endswith("wav"):
            import emotion_extract
            emo = torch.FloatTensor(emotion_extract.extract_wav(emotion))
        else:
            print("emotion参数不正确")

        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=1, emo=emo)[0][0,0].data.float().numpy()
    with open(f"../diff-svc/raw/temp_{global_steps}.wav", 'wb') as f:
        f.write(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False).data)
    print("-", end="")
    playsound.playsound(f"../diff-svc/raw/temp_{global_steps}.wav")
while True:
    text = input("Text:")
    fl = False
    for x in ['JA', 'EN', 'ZH']:
        if x in text:
            fl = True
    tts(text.split("|"
    )[0] if fl else f'[JA]{text.split("|")[0]}[JA]', "random_sample" if len(text.split("|")) < 2 else "ATRI_VD_WAV/"+text.split("|")[1], symbols_)






