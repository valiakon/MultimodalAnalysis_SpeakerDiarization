# !/usr/bin/env python3
import cv2
import imutils
import numpy as np
import pandas as pd
import face_recognition
from imutils import face_utils
from PIL import Image, ImageDraw
from visual import HOG_features

'''Cropping the rectangle with the face from the original(frame) image. Then HOG features 
are extracted from the new image containing only the face.'''
def face_features(boxes, image_np, r):
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
		#cv2.imshow("cropped", crop_img)						
		f, fn = HOG_features(crop_img)
		faces_feat.append(f)
	return faces_feat



'''Detecting faces in the median frame for every second. 
Extracting HOG features from the images containing only the faces.'''
def face_detection(path, filename):
	videopath = path + filename
	cap = cv2.VideoCapture(videopath)
	fps = cap.get(cv2.CAP_PROP_FPS)
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	fps_median = int((fps)/2)
	final_faces_feat = []
	times = 0
	next_time = fps_median
	while True:
		times +=1
		ret, image_np = cap.read()		#reading one frame per time
		if (times == next_time):
			if next_time < length:			
				next_time = times + int(fps + 1)  	#process only the median frame per second
			else:
				next_time = length
			if not image_np is None:
				print ("processing the", times, "frame")
				rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
				rgb = imutils.resize(image_np, width=750)
				r = image_np.shape[1] / float(rgb.shape[1])
				''' boxes contains the coordinates of the faces showing in frame'''
				boxes = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=2)   #select cnn model for accuracy or hog model for speed

				faces_feat = face_features(boxes, image_np, r)
				
				'''Saving HOG features for all the faces detected. If no face is detected take the features of the last person appeared 				on screen. if no face was detected in previous frames or it is the first frame and no face is detected fill zero value for 					each place in the vector.''' 
				if len(faces_feat) == 1:
					x = faces_feat[0]
					final_faces_feat.append([x])     #for each second appending a new list of features
				elif len(faces_feat) > 1:
					for i in range(len(faces_feat)):
						if i == 0:
							x = faces_feat[i]
						else:
							x = x + faces_feat[i]	
					final_faces_feat.append([x])
				else:
					if (times > 70):
						x = final_faces_feat[-1]
					else:
						x = [0 for i in range(1024)]
					final_faces_feat.append([x])     

				'''image_np = cv2.resize(image_np, dsize=(350,200))		#show frame for debugging
				new_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
				cv2.imshow('Face Detection', new_image)'''

			else:
				break

			'''if cv2.waitKey(40) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break'''

	return final_faces_feat[:55]
