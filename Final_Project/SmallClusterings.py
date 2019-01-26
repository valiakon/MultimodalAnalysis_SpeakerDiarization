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


def main():
    
    audio_path = '/home/eleni/Desktop/Multimodal/pyAudioAnalysis/pyAudioAnalysis/'   #'/Users/thanasiskaridis/Desktop/multimodal_/full_videos/'
    files = os.listdir(audio_path)
    face_path = '/home/eleni/Desktop/Multimodal/Data/Face/'
    mouth_path = '/home/eleni/Desktop/Multimodal/Data/Mouth/'
    for f in files:
        if f[-10:] == "-audio.csv":
            write_line = []
            write_line_visual = []
            write_line_all = []
            name_list = [f[:-4]+"CL0", f[:-4]+"CL1", f[:-4]+"CL2"]

            audio_feats = pd.read_csv(f, sep = ',', header=None, index_col=False)
            face_file = list_files(face_path, f[:-10]+'-face.csv')
            face_feats = pd.read_csv(face_path+face_file[0], sep = ',', header=None, index_col=False)
            mouth_file = list_files(mouth_path, f[:-10]+'-mouth.csv')
            mouth_feats = pd.read_csv(mouth_path+mouth_file[0], sep = ',', header=None, index_col=False)
            
            x = mouth_feats.values #normalizing the mouth features / What about face ones???
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            mouth_feats = pd.DataFrame(x_scaled)

            spemb = manifold.SpectralEmbedding(n_components=15, affinity = 'rbf')
            manifold_15Da = spemb.fit_transform(audio_feats)
            manifold_15Audio = pd.DataFrame(manifold_15Da)

            kmeans_audio = KMeans(n_clusters=3, random_state=0).fit(manifold_15Audio)
            labels_audio = kmeans_audio.predict(manifold_15Audio)
            print "Audio Sillhouette", metrics.silhouette_score(manifold_15Audio, labels_audio, metric='euclidean')

            indexing = [[i for i, e in enumerate(labels_audio) if e == 0], [i for i, e in enumerate(labels_audio) if e == 1], [i for i, e in enumerate(labels_audio) if e == 2]]
            centers = kmeans_audio.cluster_centers_
            write_line.append(centers)
            df = pd.DataFrame(np.row_stack(write_line))
            df['Origin'] = name_list
#            fullStr = ' '.join(str(labels.tolist()))
            df['Labels'] = indexing
            df.to_csv('/home/eleni/Desktop/Multimodal/Results/audio_clustering_results.csv', sep = ',', mode= 'a', index=False, header=False)

            face_mouth_feats = pd.concat([face_feats, mouth_feats], axis=1, ignore_index=True)
            kmeans_visual = KMeans(n_clusters=3, random_state=0).fit(face_mouth_feats)
            labels_visual = kmeans_visual.predict(face_mouth_feats)
            print "Visual Sillhouette", metrics.silhouette_score(face_mouth_feats, labels_visual, metric='euclidean') 

            indexing_visual = [[i for i, e in enumerate(labels_visual) if e == 0], [i for i, e in enumerate(labels_visual) if e == 1], [i for i, e in enumerate(labels_visual) if e == 2]]
            centers_visual = kmeans_visual.cluster_centers_
            write_line_visual.append(centers_visual)
            df_visual = pd.DataFrame(np.row_stack(write_line_visual))
            df_visual['Origin'] = name_list
#            fullStr = ' '.join(str(labels.tolist()))
            df_visual['Labels'] = indexing_visual
            df_visual.to_csv('/home/eleni/Desktop/Multimodal/Results/visual_clustering_results.csv', sep = ',', mode= 'a', index=False, header=False)

            all_feats = pd.concat([audio_feats, face_feats, mouth_feats], axis=1, ignore_index=True)
            kmeans_all = KMeans(n_clusters=3, random_state=0).fit(all_feats)
            labels_all = kmeans_all.predict(all_feats)
            print "All Sillhouette", metrics.silhouette_score(all_feats, labels_all, metric='euclidean')

            indexing_all = [[i for i, e in enumerate(labels_all) if e == 0], [i for i, e in enumerate(labels_all) if e == 1], [i for i, e in enumerate(labels_all) if e == 2]]
            centers_all = kmeans_all.cluster_centers_
            write_line_all.append(centers_all)
            df_all = pd.DataFrame(np.row_stack(write_line_all))
            df_all['Origin'] = name_list
#            fullStr = ' '.join(str(labels.tolist()))
            df_all['Labels'] = indexing_visual
            df_all.to_csv('/home/eleni/Desktop/Multimodal/Results/all_clustering_results.csv', sep = ',', mode= 'a', index=False, header=False)
           

 
if __name__== "__main__":
  main()

