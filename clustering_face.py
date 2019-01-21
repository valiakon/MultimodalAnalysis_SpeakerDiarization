import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

if __name__ == '__main__':

def FaceClustering():

	feat = pd.read_csv('visual_features_HOG.csv', delimiter=',', header=None)
	
#	kmeans = KMeans(n_clusters=3, random_state=0).fit_predict(feat)
#	print "kmeans", kmeans
	
#	agglo =  AgglomerativeClustering(n_clusters = 3,  affinity='euclidean').fit_predict(feat)
#	print "agglom", agglo

	spect = SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=0).fit_predict(feat)
	print "spect", spect

    return metrics.silhouette_score(feat, spect, metric='euclidean')
