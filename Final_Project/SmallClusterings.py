import os
import glob
import operator
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import manifold
from sklearn import preprocessing
from sklearn.cluster import KMeans


def list_files(directory, f):
    saved = os.getcwd()
    os.chdir(directory)
    it = glob.glob(f)
    os.chdir(saved)
    return it

def clustering(f, data, name_list):

    write_line = []
    kmeans = KMeans(n_clusters=3, random_state=0)
    labels = kmeans.fit_predict(data)
    silhouette_val = metrics.silhouette_score(data, labels, metric='euclidean')
    indexing = [[i for i, e in enumerate(labels) if e == 0], [i for i, e in enumerate(labels) if e == 1], [i for i, e in enumerate(labels) if e == 2]]
    centers = kmeans.cluster_centers_
    write_line.append(centers)
    df = pd.DataFrame(np.row_stack(write_line))
    df['Origin'] = name_list
    df['Labels'] = indexing
    if f[-9:-4] == 'audio':
        df.to_csv('./Run/Partial_Results/audio_clustering_results.csv', sep = ',', mode= 'a', index=False, header=False)
    elif f[-8:-4] == 'face':
        df.to_csv('./Run/Partial_Results/visual_clustering_results.csv', sep = ',', mode= 'a', index=False, header=False)
    else:
        df.to_csv('./Run/Partial_Results/all_clustering_results.csv', sep = ',', mode= 'a', index=False, header=False)
    return silhouette_val

def main():
    
    audio_path = './Run/Audio/'   
    files = os.listdir(audio_path)
    face_path = './Run/Face/'
    mouth_path = './Run/Mouth/'
    for f in files:
        if f[-10:] == "-audio.csv":
            name_list_audio = [f[:-4]+"CL0", f[:-4]+"CL1", f[:-4]+"CL2"]
            name_list_visual = [f[:-9]+"visualCL0", f[:-9]+"visualCL1", f[:-9]+"visualCL2"]
            name_list_all = [f[:-9]+"allCL0", f[:-9]+"allCL1", f[:-9]+"allCL2"]

            audio_file = list_files(audio_path, f[:-10]+'-audio.csv')
            audio_feats = pd.read_csv(audio_path+audio_file[0], sep = ',', header=None, index_col=False)
            face_file = list_files(face_path, f[:-10]+'-face.csv')
            face_feats = pd.read_csv(face_path+face_file[0], sep = ',', header=None, index_col=False)
            mouth_file = list_files(mouth_path, f[:-10]+'-mouth.csv')
            mouth_feats = pd.read_csv(mouth_path+mouth_file[0], sep = ',', header=None, index_col=False)

            audio_sil = clustering(f, audio_feats, name_list_audio)

            visual_feats = pd.concat([face_feats, mouth_feats], axis=1, ignore_index=True)
            visual_sil = clustering(face_file[0], visual_feats, name_list_visual)

            all_feats = pd.concat([audio_feats, face_feats, mouth_feats], axis=1, ignore_index=True)
            all_sil = clustering(mouth_file[0], all_feats, name_list_all)
            
            print ("Audio Silhouette:", audio_sil)
            print ("Visual Silhouette:", visual_sil)
            print ("All Silhouette:", all_sil)

           
if __name__== "__main__":
  main()

