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
from imutils import face_utils


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
        print ("encoding image", i, "..")
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
    print ("serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open("/home/valia/Desktop/encodings", "wb")
    f.write(pickle.dumps(data))
    f.close()
        

    return data

def videoFaceRecognition(data):

    print("opening video...")
    all_faces_in_video = ["Johnny", "Ellen"]
    vid = av.open('/home/valia/Desktop/cut1.mp4')
    #out = cv2.VideoWriter('/home/valia/Desktop/outputVideo.avi	', -1, 20.0, (640,480))
    print ("matching encodings and detecting faces...")
    for packet in vid.demux():
        for frame in packet.decode():
           
            x = str(type(frame))
            #print (x)
            if(x == "<class 'av.video.frame.VideoFrame'>"):
                img = frame.to_image()  
                arr = np.asarray(img)  
               # cv2.imshow("frame", arr)
               # cv2.waitKey(0)
                rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                rgb = imutils.resize(arr, width=750)
                r = arr.shape[1] / float(rgb.shape[1])
                #print r
               
                boxes = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=2)
                #if len(boxes)!=1:
                #print (len(boxes))
                encodings = face_recognition.face_encodings(rgb, boxes)
                names = []
		#name = ''
        # loop over the facial embeddings
                for encoding in encodings:
    		# match faces to encodings
                    matches = face_recognition.compare_faces(data["encodings"], encoding)
                    name = "Unknown"
                    
    		
                    if True in matches:
    			# find the best match
                        matched = [i for (i, b) in enumerate(matches) if b]
                        counts = {}
                        #print (matched)
                        #print ("in matches")
                        for i in matched:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1
                        
                        name = max(counts, key=counts.get)
                        #print (counts.getkey)
    		# maybe maax has to be changed with sth more relaxed
                   
                    names.append(name)
                    print (names)
                    
                if "Unknown" in names:
                    #names.remove('Unknown')
                    for index, namein in enumerate(names):
                        print (index)
                        if names[index] == 'Unknown':
                            temp = [x for x in all_faces_in_video if x not in names]
                            names[index] = temp[0]
                            break
  
                print ("before drawing", names)
                for ((top, right, bottom, left), name) in zip(boxes, names):
   		 # rescale the face coordinates
                    #print ("in drawinig",name)
                    top = int(top * r)
                    right = int(right * r)
                    bottom = int(bottom * r)
                    left = int(left * r)

    # draw the predicted face name on the image
                    cv2.rectangle(arr, (left, top), (right, bottom),	(0, 255, 0), 2)
                    y = top - 15 if top - 15 > 15 else top + 15 
                    cv2.putText(arr, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    #print ("image")
                cv2.imshow("frame", arr)
                cv2.waitKey(100)
		    
                    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    #out = cv2.VideoWriter('/home/valia/Desktop/outputVideo.avi',fourcc, 20.0, (arr.shape[1],arr.shape[0])) #ToDo see how to save video
	# if the writer is not None, write the frame with recognized
	# faces to disk
            	     
    print ("finished recognition...")
    #return names, boxes
    

    
def main():
    videoPaths = findVideos()
  
  
    imagePaths = findImages()
    
   
    createEncodings(imagePaths)

    data = pickle.loads(open("/home/valia/Desktop/encodings", "rb").read())
    #print len(data)
    videoFaceRecognition(data)

        
        
if __name__== "__main__":
  main()
