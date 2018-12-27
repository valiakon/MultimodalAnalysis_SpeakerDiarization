import numpy as np
import cv2
import imutils
import av
import collections
import face_recognition
from PIL import Image, ImageDraw

def findVideos():
    videoList = []
    for root, dirs, files in os.walk("/home/valia/Desktop/all_videos"):
        for file in files:
            if file.endswith(".mp4"):
                videoList.append(os.path.join(root, file))
    return (videoList)

def visualize_facial_landmarks(shape = (48, 68), color = (0,0,255)):
    vid = av.open('/home/valia/Desktop/videos/obamaSmall.mp4')
    FACIAL_LANDMARKS_IDXS = collections.OrderedDict([("mouth", (48, 68))])
    times = 0
    for packet in vid.demux():
        for frame in packet.decode():
            x = str(type(frame))
	    print x
            if(x == "<type 'av.video.frame.VideoFrame'>"):
                img = frame.to_image()  
                arr = np.asarray(img)
		pil_image = Image.fromarray(arr)
    		d = ImageDraw.Draw(pil_image, 'RGBA')
		#cv2.imshow("original", arr)
		#cv2.waitKey(0)
    		face_landmarks_list = face_recognition.face_landmarks(arr)
		i = 0
		speaker1 = []
		speaker2 = []
		for face_landmarks in face_landmarks_list:
			i = i + 1
			#d.polygon(face_landmarks['top_lip'], fill=(0, 255, 0, 128))
    			#d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
    			d.line(face_landmarks['top_lip'], fill=(0, 255, 0, 128), width=5)
    			d.line(face_landmarks['bottom_lip'], fill=(0, 255, 0, 128), width=5)
			if i == 1:
				speaker1 = [face_landmarks['top_lip'], face_landmarks['bottom_lip']]
			else:
				speaker2 = [face_landmarks['top_lip'], face_landmarks['bottom_lip']]

		if speaker2:			
			if not (speaker1[0][0] < speaker2[0][0]):
				temp = speaker1
				speaker1 = speaker2
				speaker2 = temp
		#print "speaker1", speaker1
		#print "speaker2", speaker2
		times = times + 1
		distance1 = 0
		distance2 = 0
		if times != 1:
			if speaker1:
				distance_height1 = abs(speaker1[1][2][1] - speaker1[0][2][1]) + abs(speaker1[1][3][1] - speaker1[0][3][1]) + abs(speaker1[1][4][1] - speaker1[0][4][1]) + abs(speaker1[1][5][1] - speaker1[0][5][1]) + abs(speaker1[1][6][1] - speaker1[0][6][1]) + abs(speaker1[1][7][1] - speaker1[0][7][1]) + abs(speaker1[1][8][1] - speaker1[0][8][1]) + abs(speaker1[1][9][1] - speaker1[0][9][1])

				previous_distance1 = abs(previousLipsSpeaker1[1][2][1] - previousLipsSpeaker1[0][2][1]) + abs(previousLipsSpeaker1[1][3][1] - previousLipsSpeaker1[0][3][1]) + abs(previousLipsSpeaker1[1][4][1] - previousLipsSpeaker1[0][4][1]) + abs(previousLipsSpeaker1[1][5][1] - previousLipsSpeaker1[0][5][1]) + abs(previousLipsSpeaker1[1][6][1] - previousLipsSpeaker1[0][6][1]) + abs(previousLipsSpeaker1[1][7][1] - previousLipsSpeaker1[0][7][1]) + abs(previousLipsSpeaker1[1][8][1] - previousLipsSpeaker1[0][8][1]) + abs(previousLipsSpeaker1[1][9][1] - previousLipsSpeaker1[0][9][1])		#toDo maybe try to find the emvadon of the area in the future
				
				distance1 = distance_height1 - previous_distance1
				#print "distance1 ",distance1
			if speaker2 and previousLipsSpeaker2:
				distance_height2 = abs(speaker2[1][2][1] - speaker2[0][2][1]) + abs(speaker2[1][3][1] - speaker2[0][3][1]) + abs(speaker2[1][4][1] - speaker2[0][4][1]) + abs(speaker2[1][5][1] - speaker2[0][5][1]) + abs(speaker2[1][6][1] - speaker2[0][6][1]) + abs(speaker2[1][7][1] - speaker2[0][7][1]) + abs(speaker2[1][8][1] - speaker2[0][8][1]) + abs(speaker2[1][9][1] - speaker2[0][9][1])
			
				previous_distance2 = abs(previousLipsSpeaker2[1][2][1] - previousLipsSpeaker2[0][2][1]) + abs(previousLipsSpeaker2[1][3][1] - previousLipsSpeaker2[0][3][1]) + abs(previousLipsSpeaker2[1][4][1] - previousLipsSpeaker2[0][4][1]) + abs(previousLipsSpeaker2[1][5][1] - previousLipsSpeaker2[0][5][1]) + abs(previousLipsSpeaker2[1][6][1] - previousLipsSpeaker2[0][6][1]) + abs(previousLipsSpeaker2[1][7][1] - previousLipsSpeaker2[0][7][1]) + abs(previousLipsSpeaker2[1][8][1] - previousLipsSpeaker2[0][8][1]) + abs(previousLipsSpeaker2[1][9][1] - previousLipsSpeaker2[0][9][1])

				distance2 = distance_height2 - previous_distance2
				#print "distance2 ",distance2
		
			if speaker1 and speaker2:
				if abs(distance1) > abs(distance2):
					print "Speaker1 is speaking"
				elif abs(distance2) > abs(distance1):
					print "Speaker2 is speaking"
				elif distance1 == 0 and distance2 ==0:
					print "None is speaking"
			if speaker1 and not speaker2:
				if abs(distance1) > 4:			#toDo maybe the user must type a threshold value
					print "Speaker1 is speaking"
		
		
		if speaker1:
			previousLipsSpeaker1 = speaker1 		
			if speaker2:
				previousLipsSpeaker2 = speaker2
			else:
				previousLipsSpeaker2 = []
		if(x == "<type 'av.audio.frame.AudioFrame'>"):
			print "audio"
			
		pix = np.array(pil_image)
		pix = cv2.cvtColor(pix, cv2.COLOR_RGBA2BGR)
		pix = cv2.resize(pix, dsize=(750,500))
		cv2.imshow("lips", pix)
		cv2.waitKey(10)

def main():
    visualize_facial_landmarks()

if __name__== "__main__":
    main()


