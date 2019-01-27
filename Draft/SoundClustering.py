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
from audioSegmentation import speakerDiarization
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_classif
from FeatureExtraction_Eleni import FeaturesFromAllVideos

def main():
    

    all_feats = FeaturesFromAllVideos()

    spemb = manifold.SpectralEmbedding(n_components=15, affinity = 'rbf')
    manifold_2Da = spemb.fit_transform(all_feats)
    manifold_2D = pd.DataFrame(manifold_2Da)

    hc = AgglomerativeClustering(n_clusters=2, compute_full_tree='False', affinity = 'euclidean', linkage = 'ward')
    labels = hc.fit_predict(all_feats)

    print metrics.silhouette_score(all_feats, labels, metric='euclidean')


#    print len(annot)

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
   
 


# create dendrogram
#    dendrogram = sch.dendrogram(sch.linkage(manifold_2D, method='ward'))
# create clusters
#    sc = SpectralClustering(n_clusters=2, affinity='rbf')
# save clusters for chart
#    labels = sc.fit_predict(manifold_10D)
#    print labels
#    print metrics.silhouette_score(manifold_10D, labels)
#    print metrics.completeness_score(labels, annot)

    
# save clusters for chart
    
#    print labels
    
#    print metrics.completeness_score(labels, annot)

#    model = sklearn.cluster.KMeans(n_clusters=2)
#    labels = model.fit_predict(manifold_10D)
#    print labels
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

