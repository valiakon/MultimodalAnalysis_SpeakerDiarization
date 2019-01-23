# !/usr/bin/env python3

import pandas as pd
from sklearn import metrics
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering


def FaceMouthClustering():
_
	face_feats = pd.read_csv('final_face_features.csv', delimiter=',', header=None)
    mouth_feats = pd.read_csv('final_mouth_features.csv', delimiter=',', header=None)
    sp_feats = pd.read_csv("final_audio_features.csv", sep = ',', header=None)

    manifold15 = manifold.SpectralEmbedding(n_components=15, affinity = 'rbf')
    manifold_15Da = manifold15.fit_transform(sp_feats)
    speech_feats = pd.DataFrame(manifold_15Da)

    all_feats = pd.concat([face_feats, mouth_feats, speech_feats], axis=1, ignore_index=True)
	
#	kmeans = KMeans(n_clusters=3, random_state=0).fit_predict(all_feats)
#	print "kmeans", kmeans
	
#	agglo =  AgglomerativeClustering(n_clusters = 3,  affinity='euclidean').fit_predict(all_feats)
#	print "agglom", agglo

	spect = SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=0).fit_predict(all_feats)
#	print "Spectral Clustering: ", spect

	return metrics.silhouette_score(all_feats, spect, metric='euclidean')
