import cv2
from imutils import face_utils
import face_recognition
import imutils
import os
import collections
from collections import OrderedDict
from PIL import Image, ImageDraw
import numpy as np

if __name__ == '__main__':

	cap = cv2.VideoCapture('/home/valia/Desktop/videos/obama22.mp4')
	#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
	#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

	fps = cap.get(cv2.CAP_PROP_FPS)
	times = 0
	distance = []
	while True:
		ret, image_np = cap.read()
		FACIAL_LANDMARKS_IDXS = collections.OrderedDict([("mouth", (48, 68))])
		if not image_np is None:
			times += 1
			face_landmarks_list = face_recognition.face_landmarks(image_np)
			arr = np.asarray(image_np)
			#print face_landmarks_list
			pil_image = Image.fromarray(arr)
			new_image = ImageDraw.Draw(pil_image, 'RGBA')
			speaker_number = 0
			speakers_lips = []
			for face_landmarks in face_landmarks_list:
				speaker_number += 1
				#print len(face_landmarks['top_lip'])
				new_image.line(face_landmarks['top_lip'], fill=(0, 255, 0), width=2)
				new_image.line(face_landmarks['bottom_lip'], fill=(0, 255, 0), width=2)
				speakers_lips.append([face_landmarks['top_lip'], face_landmarks['bottom_lip']])
			#print len(speakers_lips)
			
			if len(speakers_lips) > 1:
				for i, x in enumerate(speakers_lips):
					#print speakers_lips[i][0]
					if speakers_lips[i] != speakers_lips[-1]:					
						if speakers_lips[i][0][0] > speakers_lips[i+1][0][0]:
							#print speakers_lips[i][0][0]
							temp = speakers_lips[i+1]
							speakers_lips[i] = speakers_lips[i+1]
							speakers_lips[i+1] = temp

			print speakers_lips
			
			if times != 1 and (len(pr_speakers_lips) == len(speakers_lips)):
				print 'times', times
				if len(speakers_lips) >= 1:
					pr_dist = 0
					dist_temp = []
					print len(speakers_lips)
					for k in range (len(speakers_lips)):
						dist = 0
						for j in range(2, 10):
							diff = abs(speakers_lips[k][0][j][1] - speakers_lips[k][1][j][1])
							dist = dist + diff
							pr_diff = abs(pr_speakers_lips[k][0][j][1] - pr_speakers_lips[k][1][j][1])
							pr_dist = pr_dist + pr_diff
						dist_temp.append(abs(dist - pr_dist))
						if k == 1:
							distance.append(dist_temp)
							dist_temp = []
						else:
							continue
					if k != 1:
						distance.append(dist_temp)
						dist_temp = []						
			print "final distance", distance
			pr_speakers_lips = speakers_lips
			
			if len(distance):
				for i in range(len(distance)):
					pass
			
			if times % (fps + 1) == 0:
				distance = []
			print speakers_lips
			pix = np.array(pil_image)
			pix = cv2.cvtColor(pix, cv2.COLOR_RGBA2RGB)
			pix = cv2.resize(pix, dsize=(750,500))
			cv2.imshow("lips", pix)
			cv2.waitKey(10)
		else:
			break
		
		if cv2.waitKey(40) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
	
