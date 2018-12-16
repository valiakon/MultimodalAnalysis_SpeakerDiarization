# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import os
import face_recognition
import pickle
import imutils
import av
import numpy as np


def findVideos():
    videoList = []
    for root, dirs, files in os.walk("/home/valia/Desktop/all_videos"):
        for file in files:
            if file.endswith(".mp4"):
                videoList.append(os.path.join(root, file))
    return (videoList)
                
                
def findImages():
    imageList = []

    for root, dirs, files in os.walk("/home/valia/Desktop/images"):
        for file in files:
            if file.endswith(".jpeg"):
                imageList.append(os.path.join(root, file))
            
    return (imageList)
                

def createEncodings(imagePaths):
    knownEncodings = []
    knownNames = []
    for (i, imagePath) in enumerate(imagePaths):	
    	print "encoding image", i, ".."
        if imagePath[27] == 'j':
            name = "Johnny"
        elif imagePath[27] == 'e':
            name = "Ellen"
    
  
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")  
        
        encodings = face_recognition.face_encodings(rgb, boxes)
    
	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

    # write encodings to pickle file
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open("/home/valia/Desktop/encodings", "wb")
    f.write(pickle.dumps(data))
    f.close()
        

    return data

def videoFaceRecognition(data):

    print("opening video...")
    
    vid = av.open('/home/valia/Desktop/all_videos/johnny2.mp4')
 
    for packet in vid.demux():
        for frame in packet.decode():
            x = str(type(frame))
            if(x == "<type 'av.video.frame.VideoFrame'>"):
                img = frame.to_image()  
                arr = np.asarray(img)  
               # cv2.imshow("frame", arr)
               # cv2.waitKey(0)
                rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                rgb = imutils.resize(arr, width=750)
                r = arr.shape[1] / float(rgb.shape[1])
                #print r
                
                boxes = face_recognition.face_locations(rgb, model="hog")
                encodings = face_recognition.face_encodings(rgb, boxes)
                names = []
        # loop over the facial embeddings
                for encoding in encodings:
    		# match faces to encodings
                    matches = face_recognition.compare_faces(data["encodings"], encoding)
                    name = "Unknown"
     
    		
                    if True in matches:
    			# find the best match
                        matched = [i for (i, b) in enumerate(matches) if b]
                        counts = {}

                        for i in matched:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1

                        name = max(counts, key=counts.get)
    		
    		# maybe maax has to be changed with sth more relaxed
                        names.append(name)
                print names
    

        
    
def main():
    videoPaths = findVideos()
  
  
    imagePaths = findImages()
    
   
    createEncodings(imagePaths)

    data = pickle.loads(open("/home/valia/Desktop/encodings", "rb").read())
 
    videoFaceRecognition(data)

        
        
if __name__== "__main__":
  main()