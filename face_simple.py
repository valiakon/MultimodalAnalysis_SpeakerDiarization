import cv2
import face_recognition
from imutils import face_utils
import imutils
import os
import pickle
from collections import OrderedDict


if __name__ == '__main__':
	
    cap = cv2.VideoCapture('/home/valia/Desktop/videos/obama22.mp4')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print 'fps = ', fps
    fps_median = int(fps)/2
    print fps_median 
    facesInFrame = []
    times = 0
    while True:
		times +=1
		ret, image_np = cap.read()
		print times
		if (times == fps_median) or ((times % (int(fps_median + fps))) == 0):
			print times
			if not image_np is None:
				print times
				image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
				rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
				rgb = imutils.resize(image_np, width=750)
				r = image_np.shape[1] / float(rgb.shape[1])
				boxes = face_recognition.face_locations(rgb, model="cnn", number_of_times_to_upsample=2)
				print boxes
				
				for (top, right, bottom, left) in boxes:
					top = int(top * r)
					right = int(right * r)
					bottom = int(bottom * r)
					left = int(left * r)
					cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)
					y = top - 15 if top - 15 > 15 else top + 15
					#cv2.putText(image_np, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
				facesInFrame.append(len(boxes))

				image_np = cv2.resize(image_np, dsize=(350,200))
				cv2.imshow('Mouth Detection', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
			else:
				break

			if cv2.waitKey(40) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break
