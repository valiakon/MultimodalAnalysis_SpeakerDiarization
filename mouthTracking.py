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

def visualize_facial_landmarks(shape = (48, 68), color = (158, 163, 32), alpha=0.75):
    vid = av.open('/home/valia/Desktop/all_videos/johnnysmall.mp4')
    FACIAL_LANDMARKS_IDXS = collections.OrderedDict([("mouth", (48, 68))])
    for packet in vid.demux():
        for frame in packet.decode():
	    times = 0
            x = str(type(frame))
            if(x == "<type 'av.video.frame.VideoFrame'>"):
                img = frame.to_image()  
                arr = np.asarray(img)
		pil_image = Image.fromarray(arr)
    		d = ImageDraw.Draw(pil_image, 'RGBA')
		cv2.imshow("original", arr)
		cv2.waitKey(0)
    		face_landmarks_list = face_recognition.face_landmarks(arr)
		for face_landmarks in face_landmarks_list:
			#mouth = [face_landmarks['top_lip'], face_landmarks['bottom_lip']]
			#print mouth
			d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
    			d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
    			d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
    			d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)
		
		pix = np.array(pil_image)
		cv2.imshow("lips", pix)
		cv2.waitKey(0)
		#if times == 0:
		 #   return
def main():
    visualize_facial_landmarks()

if __name__== "__main__":
    main()


