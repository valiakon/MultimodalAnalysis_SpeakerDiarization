# MultimodalAnalysis_SpeakerDiarization
## Semi-supervised Mthod


### Speaker Diarization

The project tries to solve the problem of "who speaks when" in a video file. A semi-supervised method is implemented. Speaker Diarization is a challenging problem because audio and image have to be combined to result in better accuracy, when usually the number of persons participating is unknown. Additionally, other problems it has to face are that two or more persons could speak at the same time and both noise and background music would be a "part" of the discussion.


### Algorithm Steps

1. Create dataset and Seperate video from audio
2. Extract audio features from wav files
3. Perform lip tracking on video frames
4. Perform face detection and extract HOG features from images presenting only face
5. Clustering with k-means algorithm using audio features, video features(face, mouth) combined and seperately
6. Clustering of the above clusters for evaluation

#### Step 1

Firstly, we create a dataset containing videos of interviews. Usually, in the expirements two persons participated in each video but the project works for more persons, even though with lower accuracy results.  
To seperate the video from the audio segments, ffmpeg library is used. It creates wav files in order to be processed later for audio speaker Diarazation.

#### Step 2

Using the pyAudioAnalysis library we extract feature vectors for each second of the video. The aforementioned feature vectors contain mid term features related to Time-domain (Energy), Frequency-domain (Spectral) and Cepstral-domain features (MFCCs). For more information about audio features visit https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction.

#### Step 3

The third step is to perform a lip tracking procedure. This could be challenging becuse lips detection is not always successful regarding to the tendency of the face. By using the facial landamarks detector of the face recognition library we detect top and bottom lips coordinates for each person participating in a frame. We then compute the distance of the tom lip from the bottom lip and end up summing the distances for the frames of each second. This procedure assumes that if the distance between the lips succeed a high summary number it will be translated to "the person is speaking during this second". Also, if two persons appear in the same frames it is easier to result in "who is actually talking".

#### Step 4

An important step for video diarization is the face detection. Video diarization assumes that the person presented in a frame is the "speaking" person.  We find the middle frame of each second to perform the facial feature extraction as it is nearly impossible for the camera to change showing faces in less than a second during an interview. Now, that we have the right frame we are using opencv library in order to preprocess the image and feed it to Face Locations algorithm implemented in facial recognition library. The final output of the algorithm is the boxes containing the coordinates of the faces. We use those boxes to create a new image presenting only the face/faces(if two persons are appearing in the same frame) and extract the HOG features. Hog feature vectors are extracted using the skimage's HOG algorithm.

#### Step 5

Using the sklearn's k-means algorithm and feeding it with the audio, facial and mouth features combined and seperately we get the results of "who is speaking" of each second of the video. Using features seperately, it is possible to extract some conclusions on which method, audio-video, we achieve better results.
As the method used is semi-supervised we select a k+1 number for k-means. We select k by counting the number of the speakers participating in all videos of the dataset and add 1 for the "uknown" speakers or noise.

#### Step 6

In order to evaluate the above clustering we calculate the centroids of each clusters extracted in step 5. The next step is to perform a new k-means clustering on the aforementioned centroids. We, conclusively, want to see each step 5'th cluster containig the seconds that a specific person is talking to end up in the same (big) step 6'th cluster. 
