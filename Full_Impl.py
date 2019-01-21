# !/usr/bin/env python3
import re
import sys
import time
import os
import pipes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydub import AudioSegment
import pyAudioAnalysis.audioBasicIO
import pyAudioAnalysis.audioTrainTest as aT
import pyAudioAnalysis.audioFeatureExtraction
from pyAudioAnalysis.audioTrainTest import normalizeFeatures
from pyAudioAnalysis.audioSegmentation import speakerDiarization
from pyAudioAnalysis.audioFeatureExtraction import mtFeatureExtraction as mT
import cv2
from imutils import face_utils
import face_recognition
import imutils
import collections
from collections import OrderedDict
from PIL import Image, ImageDraw
from sklearn import metrics
from sklearn import manifold
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from face_simple import face_detection
from clustering_face import FaceClustering

def StereoToMono(wavFile):

    sound = AudioSegment.from_wav(wavFile)
    sound = sound.set_channels(1)
    sound.export((wavFile[:-4]+'_mono.wav'), format="wav")
    return (wavFile[:-4]+'_mono.wav')

def video_to_audio(fileName):
    file, file_extension = os.path.splitext(fileName)
    file = pipes.quote(file)
    os.system('ffmpeg -i ' + file + file_extension + ' ' + file + '.wav')

def ExtractFeatures(newPath):
    [fs, x] = audioBasicIO.readAudioFile(newPath)
    mt_size, mt_step, st_win = 1, 1, 0.5
    [mt_feats, st_feats, _] = mT(x, fs, mt_size * fs, mt_step * fs,
                                round(fs * st_win), round(fs * st_win * 0.5))
    (mt_feats_norm, MEAN, STD) = normalizeFeatures([mt_feats.T])
    mt_feats_norm = mt_feats_norm[0]
    mt_feats_normal = mt_feats_norm[:55]
    return mt_feats_normal

def mouthDetection(path, fileName):

    videoPath = path + fileName
    print (videoPath)
    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    times = 0
    distance = []
    final_list = []
    fps_median = int((fps)/2)
    next_time = fps_median
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, image_np = cap.read()
        FACIAL_LANDMARKS_IDXS = collections.OrderedDict([("mouth", (48, 68))])
        times += 1
        if not image_np is None:
            face_landmarks_list = face_recognition.face_landmarks(image_np)
            arr = np.asarray(image_np)
            pil_image = Image.fromarray(arr)
            new_image = ImageDraw.Draw(pil_image, 'RGBA')
            speaker_number = 0
            speakers_lips = []
            for face_landmarks in face_landmarks_list:
                speaker_number += 1
                new_image.line(face_landmarks['top_lip'], fill=(0, 255, 0), width=2)
                new_image.line(face_landmarks['bottom_lip'], fill=(0, 255, 0), width=2)
                speakers_lips.append([face_landmarks['top_lip'], face_landmarks['bottom_lip']])
            
            if len(speakers_lips) > 1:
                for i, x in enumerate(speakers_lips):
                    if speakers_lips[i] != speakers_lips[-1]:                   
                        if speakers_lips[i][0][0] > speakers_lips[i+1][0][0]:
                            temp = speakers_lips[i+1]
                            speakers_lips[i] = speakers_lips[i+1]
                            speakers_lips[i+1] = temp
            print ('times', times)
            if times != 1 and (len(pr_speakers_lips) == len(speakers_lips)):
                if len(speakers_lips) >= 1:
                    pr_dist = 0
                    dist_temp = []
                    for k in range (len(speakers_lips)):
                        dist = 0
                        for j in range(2, 10):
                            diff = abs(speakers_lips[k][0][j][1] - speakers_lips[k][1][j][1])
                            dist = dist + diff
                            pr_diff = abs(pr_speakers_lips[k][0][j][1] - pr_speakers_lips[k][1][j][1])
                            pr_dist = pr_dist + pr_diff
                        dist_temp.append(abs(dist - pr_dist))
                        if len(speakers_lips) == 1:
                            dist_temp.append(0)
                            distance.append(dist_temp)
                            dist_temp = []
                        else:
                            continue
                    if len(speakers_lips) != 1:
                        distance.append(dist_temp)
                        dist_temp = []                      
            pr_speakers_lips = speakers_lips
            
            if times == next_time:
                print ('distance', len(distance))
                sum_1 = 0 
                sum_2 = 0
                for i in range(len(distance)):              
                    sum_1 = sum_1 + distance[i][0]
                    sum_2 = sum_2 + distance[i][1]
                final_list.append((sum_1, sum_2))
                distance = []
                next_time = times + int(fps + 1)
                if next_time > length:
                    next_time = length
                print (next_time)
            pix = np.array(pil_image)
            pix = cv2.cvtColor(pix, cv2.COLOR_RGBA2RGB)
            pix = cv2.resize(pix, dsize=(750,500))
            cv2.imshow("lips", pix)
            cv2.waitKey(10)
        else:
            break
        if cv2.waitKey(40) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    print (final_list, len(final_list))
    return final_list[:55]

def SpeeechClustering():
    
    all_feats = pd.read_csv("final_audio_features.csv", sep = ',')
    spemb = manifold.SpectralEmbedding(n_components=15, affinity = 'rbf')
    manifold_2Da = spemb.fit_transform(all_feats)
    manifold_2D = pd.DataFrame(manifold_2Da)
    hc = AgglomerativeClustering(n_clusters=2, compute_full_tree='False', affinity = 'euclidean', linkage = 'ward')
    labels = hc.fit_predict(all_feats)
    return metrics.silhouette_score(all_feats, labels, metric='euclidean')

def main():
    path = '/home/valia/Desktop/videos'   #'/Users/thanasiskaridis/Desktop/multimodal_/full_videos/'
    files = os.listdir(path)
    audio_features = []
    mouth_features = []
    face_features = []
    for f in files:
        if f[-3:] == "mp4":
            video_to_audio(f)
            wavFile = f[:-3]+'wav' 
            monoWav = StereoToMono(wavFile)
            mt_feats_normal = ExtractFeatures(monoWav)
            writeToCsv(audio_features, mt_feats_normal)
            for sec_features in mt_feats_normal:
                audio_features.append(sec_features)
            feats = mouthDetection(path, f)      
            print (feats )			
            for sec_features in feats:
                mouth_features.append(sec_features)
            face_feat = face_detection(path, f)
            for sec_features in face_feat:
                face_features.append(sec_features)
    final_audio_features = pd.DataFrame(np.row_stack(audio_features))
    final_audio_features.to_csv('final_audio_features.csv', sep = ',')
    final_mouth_features = pd.DataFrame(np.row_stack(mouth_features))
    final_mouth_features.to_csv('final_mouth_features.csv', sep = ',')

    shil = FaceClustering()
    print (shil)
if __name__ == '__main__':
    main()


