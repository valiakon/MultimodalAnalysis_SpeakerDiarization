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
	while True:
		times += 1
		ret, image_np = cap.read()
		FACIAL_LANDMARKS_IDXS = collections.OrderedDict([("mouth", (48, 68))])
		if not image_np is None:
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
				new_image.line(face_landmarks['top_lip'], fill=(0, 255, 0), width=5)
				new_image.line(face_landmarks['bottom_lip'], fill=(0, 255, 0), width=5)
				speakers_lips.append([face_landmarks['top_lip'], face_landmarks['bottom_lip']])
			print (speakers_lips)

			if len(speakers_lips) > 1:
				for i, x in enumerate(speakers_lips):
					print speakers_lips[i][0]
					if speakers_lips[i] != speakers_lips[-1]:					
						if speakers_lips[i][0][0] > speakers_lips[i+1][0][0]:
							print speakers_lips[i][0][0]
							temp = speakers_lips[i+1]
							speakers_lips[i] = speakers_lips[i+1]
							speakers_lips[i+1] = temp

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
	
