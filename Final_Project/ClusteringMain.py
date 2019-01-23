from Clusterings import SpeeechClustering
from Clusterings import FaceClustering
from Clusterings import FaceMouthClustering
from Clusterings import FaceMouthClustering
from Clusterings import AllFeaturesClustering
import pandas as pd


if __name__ == '__main__':
	
	shil_speech = SpeeechClustering()
	print "shil_speech", shil_speech

	shil_face = FaceClustering()
	print "shil_face", shil_face

	shil_mouth_face = FaceMouthClustering()
	print "shil_mouth_face", shil_mouth_face

	shil_all = AllFeaturesClustering()
	print "shil_all", shil_all
