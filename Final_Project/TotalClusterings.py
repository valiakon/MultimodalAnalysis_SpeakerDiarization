import os
import glob
import operator
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans


def main():
    
    path = '/home/eleni/Desktop/Multimodal/Results/'
    files = os.listdir(path)
    for f in files:
        features = pd.read_csv(path+f, sep = ',', header=None, index_col=False)
        features = features[features.columns[:-2]]
        kmeans = KMeans(n_clusters=3, random_state=0).fit(features)
        labels = kmeans.predict(features)
#        print "Sillhouette", metrics.silhouette_score(features, labels, metric='euclidean')
        write_line = []
        name_list = [f[:-4]+"CL0", f[:-4]+"CL1", f[:-4]+"CL2"]   
         
        indexing = [[i for i, e in enumerate(labels) if e == 0], [i for i, e in enumerate(labels) if e == 1], [i for i, e in enumerate(labels) if e == 2]]
        df = pd.DataFrame(indexing)
        df['Origin'] = name_list
        
        if f[:5] == "audio":
            df.to_csv('/home/eleni/Desktop/Multimodal/AllResults/all_audio_clustering_results.csv', sep = ',', index=False, header=False)
        elif f[:6] == "visual":
            df.to_csv('/home/eleni/Desktop/Multimodal/AllResults/all_visual_clustering_results.csv', sep = ',', index=False, header=False)
        else:
            df.to_csv('/home/eleni/Desktop/Multimodal/AllResults/all_all_clustering_results.csv', sep = ',', index=False, header=False)
           

 
if __name__== "__main__":
  main()

