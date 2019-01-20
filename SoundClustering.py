import os
import glob
import operator
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import manifold
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from FeatureExtraction_Eleni import FeaturesFromAllVideos


def main():
    
    all_feats = FeaturesFromAllVideos()

    spemb = manifold.SpectralEmbedding(n_components=15, affinity = 'rbf')
    manifold_2Da = spemb.fit_transform(all_feats)
    manifold_2D = pd.DataFrame(manifold_2Da)

    hc = AgglomerativeClustering(n_clusters=13, compute_full_tree='False', affinity = 'euclidean', linkage = 'ward')
    labels = hc.fit_predict(manifold_2D)

    print metrics.silhouette_score(manifold_2D, labels, metric='euclidean')       


if __name__== "__main__":
  main()

