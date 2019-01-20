import audioBasicIO
import audioFeatureExtraction
from audioFeatureExtraction import mtFeatureExtraction as mT
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import operator
from sklearn import metrics
from audioTrainTest import normalizeFeatures
from sklearn.feature_selection import mutual_info_classif
import os
from sklearn.ensemble import ExtraTreesClassifier
import audioFeatureExtraction as aF
from sklearn.metrics.cluster import adjusted_rand_score
import audioTrainTest as aT
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn import manifold
import glob

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


def main():
    
    all_features = []
    mt_size, mt_step, st_win = 1, 1, 0.5
    wavs = findwavs("/home/eleni/Desktop/Multimodal/test")
    for path in wavs:
        newPath = StereoToMono(path)
        mt_feats_norm = ExtractFeatures(newPath)
        for sec_features in mt_feats_norm:
            all_features.append(sec_features)
    all_feats = pd.DataFrame(np.row_stack(all_features))

    annot = [1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

#    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
#    forest.fit(all_feats, annot)
#    importances = forest.feature_importances_
#    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
#    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
#    print("Feature ranking:")

#    for f in range(all_feats.shape[1]):
#        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    
#    names = list(all_feats.columns.values)
#    res = dict(zip(names, mutual_info_classif(all_feats, annot)))
#    sorted_res = sorted(res.items(), key=operator.itemgetter(1))
#    print sorted_res

    
#    df = pd.DataFrame(annot, columns=['speaker'])    
   
 
#    iso = manifold.Isomap(n_neighbors=6, n_components=2)
#    iso.fit(all_feats)
#    manifold_10Da = iso.transform(all_feats)
#    manifold_10D = pd.DataFrame(manifold_10Da)
#    print manifold_10D


# create dendrogram
#    dendrogram = sch.dendrogram(sch.linkage(manifold_10D, method='ward'))
# create clusters
#    hc = AgglomerativeClustering(n_clusters=2, compute_full_tree='False', affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
#    labels = hc.fit_predict(manifold_10D)
#    print labels
#    df = pd.DataFrame(labels, columns=['speaker'])
#    print adjusted_rand_score(labels, annot)


#    model = sklearn.cluster.KMeans(n_clusters=2)
#    labels = model.fit_predict(manifold_10D)
#    print labels
#    df = pd.DataFrame(labels, columns=['speaker'])
#    print metrics.silhouette_samples(manifold_10D, labels, metric='euclidean')
#    print metrics.homogeneity_score(labels, annot)
#    print labels


#    finalDf = pd.concat([manifold_10D, df[['speaker']]], axis = 1)

#    fig = plt.figure(figsize = (8,8))
#    ax = fig.add_subplot(1,1,1) 
#    ax.set_xlabel('Principal Component 1', fontsize = 15)
#    ax.set_ylabel('Principal Component 2', fontsize = 15)
#    ax.set_title('Isomap', fontsize = 20)
#    targets = [0, 1, 2]
#    colors = ['r', 'b', 'g']
#    for target, color in zip(targets,colors):
#        indicesToKeep = finalDf['speaker'] == target
#        ax.scatter(finalDf.loc[indicesToKeep, 0], finalDf.loc[indicesToKeep, 1], c = color)
#    ax.legend(targets)
#    ax.grid()
#    plt.show()


if __name__== "__main__":
  main()

