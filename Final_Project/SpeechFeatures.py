# !/usr/bin/env python3

import os
import pipes
import numpy as np
import pandas as pd
from pydub import AudioSegment
import pyAudioAnalysis.audioBasicIO
from pyAudioAnalysis.audioTrainTest import normalizeFeatures
from pyAudioAnalysis.audioFeatureExtraction import mtFeatureExtraction as mT


def video_to_audio(fileName, filepath):
    file, file_extension = os.path.splitext(fileName)
    file = pipes.quote(file)
    os.system('ffmpeg -i ' + filepath + ' ' + file + '.wav')


def StereoToMono(wavFile):

    sound = AudioSegment.from_wav(wavFile)
    sound = sound.set_channels(1)
    sound.export((wavFile[:-4]+'_mono.wav'), format="wav")
    return (wavFile[:-4]+'_mono.wav')


def ExtractFeatures(newPath):
    print newPath
    [fs, x] = audioBasicIO.readAudioFile(newPath)
    mt_size, mt_step, st_win = 1, 1, 0.5
    [mt_feats, st_feats, _] = mT(x, fs, mt_size * fs, mt_step * fs,
                                round(fs * st_win), round(fs * st_win * 0.5))
    (mt_feats_norm, MEAN, STD) = normalizeFeatures([mt_feats.T])
    mt_feats_norm = mt_feats_norm[0]
    mt_feats_normal = mt_feats_norm[:55]
    return mt_feats_normal
