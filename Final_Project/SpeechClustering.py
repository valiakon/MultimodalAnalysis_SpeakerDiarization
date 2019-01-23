# !/usr/bin/env python3

import pandas as pd
from sklearn import metrics
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering


def SpeeechClustering():
    
    feats = pd.read_csv("final_audio_features.csv", sep = ',', header=None)

    manifold15 = manifold.SpectralEmbedding(n_components=15, affinity = 'rbf')
    manifold_15Da = manifold15.fit_transform(feats)
    speech_feats = pd.DataFrame(manifold_15Da)

    hc = AgglomerativeClustering(n_clusters=3, compute_full_tree='False', affinity = 'euclidean', linkage = 'ward')
    labels = hc.fit_predict(speech_feats)

#	kmeans = KMeans(n_clusters=3, random_state=0).fit_predict(manifold_15D)
#	print "kmeans", kmeans
	
#	spect = SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=0).fit_predict(manifold_15D)
#	print "Spectral Clustering: ", spect

    return metrics.silhouette_score(speech_feats, labels, metric='euclidean')


