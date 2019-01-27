import cv2
from imutils import face_utils
import face_recognition
import imutils
import os
import pickle
from collections import OrderedDict

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
	elif imagePath[27] == 'o':
	    name = "Obama"
	elif imagePath[27] == 'b':
	    name = "Bill"
     
  
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

def matchNames(rgb, boxes, data):
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

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
	    	names.append(name)
        if "Unknown" in names:
		    for index, namein in enumerate(names):
				if names[index] == 'Unknown':
					temp = [x for x in all_faces_in_video if x not in names]
					names[index] = temp[0]
					break
    return names

if __name__ == '__main__':
	
    cap = cv2.VideoCapture('/home/valia/Desktop/videos/obama22.mp4')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

    imagePaths = findImages()   
    createEncodings(imagePaths)
    data = pickle.loads(open("/home/valia/Desktop/encodings", "rb").read())
    fps = cap.get(cv2.CAP_PROP_FPS)
    print 'fps = ', fps
    names_final = []
    nameForSecond = []
    facesInFrame = []
    times = 0
    while True:
		times +=1
		ret, image_np = cap.read()

		if not image_np is None:
			image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
			rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
			rgb = imutils.resize(image_np, width=750)
			r = image_np.shape[1] / float(rgb.shape[1])
			boxes = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=2)
			print boxes
			
			names = matchNames(rgb, boxes, data)
			names_final.extend(names)
			#print names
			for ((top, right, bottom, left), name) in zip(boxes, names):
				top = int(top * r)
				right = int(right * r)
				bottom = int(bottom * r)
				left = int(left * r)
				#print name, left, top, right, bottom
			# draw the predicted face name on the image
				cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)
				y = top - 15 if top - 15 > 15 else top + 15
				cv2.putText(image_np, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
			facesInFrame.append(len(boxes))
			if times % (int(fps)) == 0:
				counts = {}
				for name in names_final:
				     counts[name] = counts.get(name, 0) + 1
				sorted_names =  OrderedDict(sorted(counts.items(), key=lambda i: i[1], reverse=True))
				sorted_names_list =list(sorted_names.keys())
				#the max number of faces detected in frames during one second 
				number = max(facesInFrame)
				name_list = []
				for i in range(number):
					if sorted_names_list[i]:
						name_list.append(sorted_names_list[i])						
				nameForSecond.append(name_list)
				print 'nameForSecond', nameForSecond
				names_final = []
				facesInFrame = []

			image_np = cv2.resize(image_np, dsize=(350,200))
			cv2.imshow('Mouth Detection', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
		else:
			
			if names_final:
				counts = {}
				for name in names_final:
					counts[name] = counts.get(name, 0) + 1
					#this is the feature for the multi clustering
				nameForSecond.append(max(counts, key=counts.get))
				print 'nameForSecond', nameForSecond
			break
		if cv2.waitKey(40) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
	


