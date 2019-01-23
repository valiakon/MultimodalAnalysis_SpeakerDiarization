# !/usr/bin/env python3

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering


def FaceMouthClustering():
_
	face_feats = pd.read_csv('final_face_features.csv', delimiter=',', header=None)
    mouth_feats = pd.read_csv('final_mouth_features.csv', delimiter=',', header=None)

    both_feats = pd.concat([face_feats, mouth_feats], axis=1, ignore_index=True)
	
#	kmeans = KMeans(n_clusters=3, random_state=0).fit_predict(feat)
#	print "kmeans", kmeans
	
#	agglo =  AgglomerativeClustering(n_clusters = 3,  affinity='euclidean').fit_predict(feat)
#	print "agglom", agglo

	spect = SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=0).fit_predict(both_feats)
#	print "Spectral Clustering: ", spect

	return metrics.silhouette_score(both_feats, spect, metric='euclidean')
