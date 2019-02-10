import os
import glob
import operator
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import manifold
from sklearn import preprocessing
from sklearn.cluster import KMeans

    
def main():
   

    colnames = ['audio_ground']
    df = pd.read_csv('GroundTruthSimple.csv', names=colnames)
    audio_ground = df['audio_ground'].tolist() 

    audio_feats = pd.read_csv('obama_evaluation-audio.csv', sep = ',', header=None, index_col=False)
    face_feats = pd.read_csv('obama_evaluation-face.csv', sep = ',', header=None, index_col=False)
    mouth_feats = pd.read_csv('obama_evaluation-mouth.csv', sep = ',', header=None, index_col=False)

    kmeans_audio = KMeans(n_clusters=3, random_state=0)
    labels_audio = kmeans_audio.fit_predict(audio_feats)
    audio_score = metrics.fowlkes_mallows_score(labels_audio, audio_ground)
    print "Audio Score:", audio_score

    visual_feats = pd.concat([face_feats, mouth_feats], axis=1, ignore_index=True)
    kmeans_visual = KMeans(n_clusters=3, random_state=0)
    labels_visual = kmeans_visual.fit_predict(visual_feats)
    visual_score = metrics.fowlkes_mallows_score(labels_visual, audio_ground)
    print "Visual Score:", visual_score

    all_feats = pd.concat([audio_feats, face_feats, mouth_feats], axis=1, ignore_index=True)
    kmeans_all = KMeans(n_clusters=3, random_state=0)
    labels_all = kmeans_all.fit_predict(all_feats)
    all_score = metrics.fowlkes_mallows_score(labels_all, audio_ground)
    print "All Score:", all_score
            
           
           
if __name__== "__main__":
  main()

