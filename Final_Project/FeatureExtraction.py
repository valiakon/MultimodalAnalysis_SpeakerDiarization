# !/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from SpeechFeatures import video_to_audio
from SpeechFeatures import StereoToMono
from SpeechFeatures import ExtractFeatures
from MouthDetection import mouthDetection
from FaceDetection import face_detection
import csv



def write_csv(features, filename):
	feat_to_csv_final = []
	with open(filename, 'a') as feat:
		wr = csv.writer(feat, delimiter=',')
		for i in features:
			feat_to_csv = []
			for j in range(len(i[0])):
				feat_to_csv.append(float('%.5f'%(i[0][j])))
			wr.writerow(feat_to_csv)

def main():
	path = './Videos/' 
	files = os.listdir(path)
	print files
	file_order = []
	for f in files:
		if f[-3:] == "mp4":
			file_order.append(f)
			audio_features = []
			mouth_features = []
			face_features = []
			filepath = path + f
			video_to_audio(f, filepath)			#split audio files from video
			wavFile = f[:-3]+'wav' 
			monoWav = StereoToMono(wavFile)				#tranform stereo to mono
			mt_feats_normal = ExtractFeatures(monoWav)

			'''Extracting audio features, temporal features of mouth tracking and visual features of images containing speaker faces'''
			for sec_features in mt_feats_normal:
				audio_features.append(sec_features)
			mouth_feats = mouthDetection(path, f)      	
			for sec_features in mouth_feats:
				mouth_features.append(sec_features)
			face_feats = face_detection(path, f)
			for sec_features in face_feats:
				face_features.append(sec_features)

			'''three final lists containing the final features for each model, face, mouth and audio'''
			final_audio_features = pd.DataFrame(np.row_stack(audio_features))
			final_audio_features = final_audio_features.round(5)
			final_mouth_features = pd.DataFrame(np.row_stack(mouth_features))

			'''Checking if all lists contain the same number of features w.r.t the seconds of the video, 
			keeping the number of instances of the smallest list if the numbers are different - left for future generalisation for more than 				two speakers'''
			'''if len(final_audio_features) != len(final_mouth_features):
				if len(final_audio_features) < len(final_mouth_features):
					final_mouth_features = final_mouth_features[:len(final_audio_features)]
				else:
					final_audio_features = final_audio_features[:len(final_mouth_features)]
			if len(face_features) != len(final_audio_features):
				if len(face_features) < len(final_audio_features):
					final_audio_features = final_audio_features[:len(face_features)]
				else:
					face_features = face_features[:len(final_audio_features)]
			
			if len(face_features) != len(final_mouth_features):
				if len(face_features) < len(final_mouth_features):
					final_mouth_features = final_mouth_features[:len(face_features)]
				else:
					face_features = face_features[:len(final_mouth_features)]'''

			'''writing all features lists to csv'''
			final_audio_features.to_csv(f[:-4]+'-audio.csv', sep = ',', mode= 'a', header=None, index=False)
			final_mouth_features.to_csv(f[:-4]+'-mouth.csv', sep = ',', mode= 'a', header=None, index=False)
			write_csv(face_features, f[:-4]+'-face.csv')
   
if __name__ == '__main__':
    main()


