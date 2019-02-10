# !/usr/bin/env python3

import cv2
import imutils
import numpy as np
import pandas as pd
import face_recognition
from imutils import face_utils
from PIL import Image, ImageDraw
from collections import OrderedDict

'''sort speakers from left to right'''
def sort_speakers(speakers_lips):
	for i, x in enumerate(speakers_lips):
		if speakers_lips[i] != speakers_lips[-1]:                   
		    if speakers_lips[i][0][0] > speakers_lips[i+1][0][0]:
		        temp = speakers_lips[i+1]
		        speakers_lips[i] = speakers_lips[i+1]
		        speakers_lips[i+1] = temp
	return speakers_lips


'''Sum the distances of lips of each speaker every second'''
def sum_distance(distance, final_list):
	sum_1 = 0 
	sum_2 = 0
	for i in range(len(distance)):              
		sum_1 = sum_1 + distance[i][0]
		sum_2 = sum_2 + distance[i][1]
	final_list.append((sum_1, sum_2))
	return final_list


'''Checking if previous frame contained one or more speakers. For the first image cannot calculate distance. 
If different number of speakers are detected for the first time continue to make the calculations for the next frame'''
def calculate_distance(pr_speakers_lips, speakers_lips, distance):
	if len(speakers_lips) >= 1:
		pr_dist = 0
		dist_temp = []
		for k in range (len(speakers_lips)):
			dist = 0
			for j in range(2, 10):
				diff = abs(speakers_lips[k][0][j][1] - speakers_lips[k][1][j][1])   #calculating the difference (top_lip[y] - bottom_lip[y]) only for y axis for the central coordinates of mouth
				dist = dist + diff
				pr_diff = abs(pr_speakers_lips[k][0][j][1] - pr_speakers_lips[k][1][j][1]) #calculating the distance in the previous frame
				pr_dist = pr_dist + pr_diff
			dist_temp.append(abs(dist - pr_dist))		#saving the distance of the lips between two frames
			if len(speakers_lips) == 1:
				dist_temp.append(0)
				distance.append(dist_temp)
				dist_temp = []
			else:
				continue
		if len(speakers_lips) != 1:
			distance.append(dist_temp)
			dist_temp = [] 
	return distance   

'''Detecting top and bottom lips in each frame. Calculate temporal features by calculating the sum of 
distances between top and bottom lip for every second'''       
def mouthDetection(path, fileName):
	videoPath = path + fileName
	cap = cv2.VideoCapture(videoPath)
	fps = cap.get(cv2.CAP_PROP_FPS)
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	times = 0
	distance = []
	final_list = []
	next_time = int(fps)

	while True:
		ret, image_np = cap.read()
		times += 1
		if not image_np is None:
			print ("processing the", times, "frame")
			face_landmarks_list = face_recognition.face_landmarks(image_np)
			arr = np.asarray(image_np)
			pil_image = Image.fromarray(arr)
			new_image = ImageDraw.Draw(pil_image, 'RGBA')
			speaker_number = 0
			speakers_lips = []
			'''saving the locations of top and bottom lips for each speaker in the frame'''
			for face_landmarks in face_landmarks_list:
				speaker_number += 1
				speakers_lips.append([face_landmarks['top_lip'], face_landmarks['bottom_lip']])
				new_image.line(face_landmarks['top_lip'], fill=(0, 255, 0), width=2)			#draw lips for debugging
				new_image.line(face_landmarks['bottom_lip'], fill=(0, 255, 0), width=2)

			if len(speakers_lips) > 1:
				speakers_lips = sort_speakers(speakers_lips) #sorting speakers from left to right

			
			if times != 1 and (len(pr_speakers_lips) == len(speakers_lips)):   
				distance = calculate_distance(pr_speakers_lips, speakers_lips, distance)	           
			pr_speakers_lips = speakers_lips 		#saving the last lips coordinates to be used as previous
            
			if times == next_time:
				final_list = sum_distance(distance,final_list)  	#save sum of distances for each speaker every second
				distance = []
				next_time = times + int(fps + 1)
				if next_time > length:
					next_time = length

			'''pix = np.array(pil_image)				#show image for debugging
			pix = cv2.cvtColor(pix, cv2.COLOR_RGBA2RGB)
			pix = cv2.resize(pix, dsize=(750,500))
			cv2.imshow("lips", pix)
			cv2.waitKey(10)'''
		else:
			break
		'''if cv2.waitKey(40) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break'''


	return final_list[:55]
