import cv2
import face_recognition
from imutils import face_utils
import imutils
import os
import pickle
from collections import OrderedDict
import visual_features
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def clustering(features):
	
	#kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
	df = pd.DataFrame(data=features)
	print df


if __name__ == '__main__':
	
	cap = cv2.VideoCapture('/home/valia/Desktop/videos/obama22.mp4')
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
	vs = visual_features.ImageFeatureExtractor()
	fps = cap.get(cv2.CAP_PROP_FPS)
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print( length )
	print 'fps = ', fps
	fps_median = int(fps)/2
	print fps_median 
	facesInFrame = []
	final_faces_feat = []
	times = 0
	next_time = fps_median
	while True:
		times +=1
		#print times
		ret, image_np = cap.read()
		if (times == next_time):
			if next_time < length:			
				next_time = times + int(fps + 1)
				print "next_time", next_time
			else:
				next_time = length
			if not image_np is None:
				print times
				image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
				rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
				rgb = imutils.resize(image_np, width=750)
				r = image_np.shape[1] / float(rgb.shape[1])
				#print "shape", image_np.shape
				boxes = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=2)
				print "boxes", boxes
				box_num = len(boxes)
				faces_feat = []
				for (top, right, bottom, left) in boxes:
					top = int(top * r)
					right = int(right * r)
					bottom = int(bottom * r)
					left = int(left * r)
					cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)
					crop_img = image_np[top: bottom, left:right]
					
					crop_img = cv2.resize(crop_img, dsize=(100,100))
					crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
					cv2.imshow("cropped", crop_img)						
					f, fn = vs.getRGBS(crop_img)
					faces_feat.append(f)
					#print len(faces_feat)
				if len(faces_feat) == 1:
						faces_feat.append(np.full((96), -1))
				final_faces_feat.append([faces_feat])
				#print len(final_faces_feat)
					
				facesInFrame.append(len(boxes))

				image_np = cv2.resize(image_np, dsize=(350,200))
				new_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
				cv2.imshow('Mouth Detection', new_image)

			else:
				break

			if cv2.waitKey(40) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break
	print len(final_faces_feat)
	print (final_faces_feat)
	clustering(final_faces_feat)
