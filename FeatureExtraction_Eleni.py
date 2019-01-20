import os
import numpy as np
import pandas as pd
import audioBasicIO
import audioTrainTest as aT
from sklearn import manifold
import audioFeatureExtraction
from pydub import AudioSegment
import matplotlib.pyplot as plt
import audioFeatureExtraction as aF
from audioTrainTest import normalizeFeatures
from audioFeatureExtraction import mtFeatureExtraction as mT


def findwavs(filePath):
    wavList = []

    for root, dirs, files in os.walk(filePath):
        for file in files:
            if file.endswith(".wav"):
                wavList.append(os.path.join(root, file))
            
    return (wavList)


def StereoToMono(wavFilePath):

    drive, path_and_file = os.path.splitdrive(wavFilePath)
    path, file_name = os.path.split(path_and_file)
    
    sound = AudioSegment.from_wav(wavFilePath)
    sound = sound.set_channels(1)
    outPath = os.path.join('/home/eleni/Desktop/Multimodal/mono_wavs/', file_name)
    sound.export(outPath, format="wav")
    return outPath


def ExtractFeatures(newPath):
    [fs, x] = audioBasicIO.readAudioFile(newPath)
    mt_size, mt_step, st_win = 1, 1, 0.5
    [mt_feats, st_feats, _] = mT(x, fs, mt_size * fs, mt_step * fs,
                                round(fs * st_win), round(fs * st_win * 0.5))
    (mt_feats_norm, MEAN, STD) = normalizeFeatures([mt_feats.T])
    mt_feats_norm = mt_feats_norm[0]
    #F, name = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
    #print np.shape(F)
    return mt_feats_norm

def FeaturesFromAllVideos():
    
    all_features = []
    
    wavs = findwavs("/home/eleni/Desktop/Multimodal/test")
    for path in wavs:
        newPath = StereoToMono(path)
        mt_feats_norm = ExtractFeatures(newPath)
#        mt_feats_normal = mt_feats_norm[4:]
        for sec_features in mt_feats_norm:
            all_features.append(sec_features)
    all_feats = pd.DataFrame(np.row_stack(all_features))
    return all_feats

