import os
import glob
import operator
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import manifold
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.feature_selection import mutual_info_classif
from FeatureExtraction_Eleni import FeaturesFromAllVideos


def main():
    

    all_feats = FeaturesFromAllVideos()
#    print all_feats.shape

    spemb = manifold.SpectralEmbedding(n_components=2, affinity = 'rbf')
    manifold_2Da = spemb.fit_transform(all_feats)
    manifold_2D = pd.DataFrame(manifold_2Da)

# create dendrogram
    dendrogram = sch.dendrogram(sch.linkage(manifold_2D, method='ward'))
# create clusters
    hc = AgglomerativeClustering(n_clusters=3, compute_full_tree='False', affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
    labels = hc.fit_predict(manifold_2D)
#    print labels
    print metrics.silhouette_score(manifold_2D, labels, metric='euclidean')
#    print metrics.completeness_score(labels, annot)

if __name__== "__main__":
  main()

