from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis.audioFeatureExtraction import mtFeatureExtraction as mT
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from pyAudioAnalysis.audioTrainTest import normalizeFeatures
from pyAudioAnalysis.audioSegmentation import flags2segs
import os
import readchar

def StereoToMono(wavFilePath):
    sound = AudioSegment.from_wav(wavFilePath)
    sound = sound.set_channels(1)
    outPath = "/home/valia/Desktop/obamamono.wav"
    sound.export(outPath, format="wav")
    return outPath


def ExtractFeatures(newPath):
    [fs, x] = audioBasicIO.readAudioFile(newPath)
    mt_size, mt_step, st_win = 1, 0.1, 0.05
    [mt_feats, st_feats, _] = mT(x, fs, mt_size * fs, mt_step * fs,
                                round(fs * st_win), round(fs * st_win * 0.5))
    (mt_feats_norm, MEAN, STD) = normalizeFeatures([mt_feats.T])
    mt_feats_norm = mt_feats_norm[0].T
    #F, name = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
    #print np.shape(F)
    return mt_feats_norm

def main():
    wavFilePath = "/home/valia/Desktop/obama.wav"
    mt_size, mt_step, st_win = 1, 0.1, 0.05
    newPath = StereoToMono(wavFilePath)
    mt_feats_norm = ExtractFeatures(newPath)
    #arr = np.asarray(F)
    k_means = KMeans(n_clusters=3)
    k_means.fit(mt_feats_norm.T)
    cls = k_means.labels_
    segs, c = flags2segs(cls, mt_step)      # convert flags to segment limits
    for sp in range(3):            # play each cluster's segment
        for i in range(len(c)):
            if c[i] == sp and segs[i, 1]-segs[i, 0] > 1:
                # play long segments of current speaker
                print(c[i], segs[i, 0], segs[i, 1])
                cmd = "avconv -i {} -ss {} -t {} temp.wav " \
                          "-loglevel panic -y".format(wavFilePath, segs[i, 0]+1,
                                                      segs[i, 1]-segs[i, 0]-1)
                os.system(cmd)
                os.system("play temp.wav -q")
                readchar.readchar()

if __name__== "__main__":
  main()


