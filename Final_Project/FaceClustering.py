# !/usr/bin/env python3

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering


def FaceClustering():

	face_feats = pd.read_csv('final_face_features.csv', delimiter=',', header=None)
	
#	kmeans = KMeans(n_clusters=3, random_state=0).fit_predict(feat)
#	print "kmeans", kmeans
	
#	agglo =  AgglomerativeClustering(n_clusters = 3,  affinity='euclidean').fit_predict(feat)
#	print "agglom", agglo

	spect = SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=0).fit_predict(face_feats)
#	print "Spectral Clustering: ", spect

	return metrics.silhouette_score(face_feats, spect, metric='euclidean')
