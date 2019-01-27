# MultimodalAnalysis_SpeakerDiarization


### Speaker Diarization

The project tries to solve the problem of "who speaks when" in a video file. It is a challenging problem because audio and image have to be combined to result in better accuracy, when usually the number of persons participating is unknown. Additionally, other problems it has to face are that two or more persons could speak at the same time and both noise and background music would be a "part" of the discussion.


### Algorithm Steps

1. Create dataset and Seperate video from audio
2. Extract audio features from wav files
3. Perform lip tracking on video frames
4. Perform face detection 
5. Extract HOG features from images presenting only face
6. Clustering with k-means algorithm using audio features, video features(face, mouth) combined and seperately
7. Clustering of the above clusters for evaluation

#### Step 1

Firstly, we create a dataset containing videos of interviews. Usually, in the expirements two persons participated in each video but the project works for more persons, even though with lower accuracy results.  
TO seperate the video from the audio segments, ffmpeg library is used. It creates wav files in order to be processed later for audio speaker Diarazation.

#### Step 2

Using the pyAudioAnalysis library we extract feature vectors for each second of the video. The aforementioned feature vectors contain mid term features related to Time-domain (Energy), Frequency-domain (Spectral) and Cepstral-domain features (MFCCs). For more information about audio features visit gitlink.

#### Step 3

The third step is to perform a lip tracking procedure. This could be challenging becuse lips detection is not always successful regarding to the tendency of the face. By using the facial landamarks detector of the face recognition library we detect top and bottom lips coordinates for each person participating in a frame. We then compute the distance of the tom lip from the bottom lip and end up summing the distances for the frames of each second. This procedure assumes that if the distance between the lips succeed a high summary number it will be translated to "the person is speaking during this second". Also, if two persons appear in the same frames it is easier to result in "who is actually talking".
