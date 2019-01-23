# !/usr/bin/env python3

import cv2
import imutils
import numpy as np
import pandas as pd
import face_recognition
from imutils import face_utils
from PIL import Image, ImageDraw
from collections import OrderedDict


def mouthDetection(path, fileName):

    videoPath = path + fileName
    print (videoPath)
    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    times = 0
    distance = []
    final_list = []
    fps_median = int((fps)/2)
    next_time = fps_median
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, image_np = cap.read()
        FACIAL_LANDMARKS_IDXS = OrderedDict([("mouth", (48, 68))])
        times += 1
        if not image_np is None:
            face_landmarks_list = face_recognition.face_landmarks(image_np)
            arr = np.asarray(image_np)
            pil_image = Image.fromarray(arr)
            new_image = ImageDraw.Draw(pil_image, 'RGBA')
            speaker_number = 0
            speakers_lips = []
            for face_landmarks in face_landmarks_list:
                speaker_number += 1
                new_image.line(face_landmarks['top_lip'], fill=(0, 255, 0), width=2)
                new_image.line(face_landmarks['bottom_lip'], fill=(0, 255, 0), width=2)
                speakers_lips.append([face_landmarks['top_lip'], face_landmarks['bottom_lip']])
            
            if len(speakers_lips) > 1:
                for i, x in enumerate(speakers_lips):
                    if speakers_lips[i] != speakers_lips[-1]:                   
                        if speakers_lips[i][0][0] > speakers_lips[i+1][0][0]:
                            temp = speakers_lips[i+1]
                            speakers_lips[i] = speakers_lips[i+1]
                            speakers_lips[i+1] = temp
            print ('times', times)
            if times != 1 and (len(pr_speakers_lips) == len(speakers_lips)):
                if len(speakers_lips) >= 1:
                    pr_dist = 0
                    dist_temp = []
                    for k in range (len(speakers_lips)):
                        dist = 0
                        for j in range(2, 10):
                            diff = abs(speakers_lips[k][0][j][1] - speakers_lips[k][1][j][1])
                            dist = dist + diff
                            pr_diff = abs(pr_speakers_lips[k][0][j][1] - pr_speakers_lips[k][1][j][1])
                            pr_dist = pr_dist + pr_diff
                        dist_temp.append(abs(dist - pr_dist))
                        if len(speakers_lips) == 1:
                            dist_temp.append(0)
                            distance.append(dist_temp)
                            dist_temp = []
                        else:
                            continue
                    if len(speakers_lips) != 1:
                        distance.append(dist_temp)
                        dist_temp = []                      
            pr_speakers_lips = speakers_lips
            
            if times == next_time:
                print ('distance', len(distance))
                sum_1 = 0 
                sum_2 = 0
                for i in range(len(distance)):              
                    sum_1 = sum_1 + distance[i][0]
                    sum_2 = sum_2 + distance[i][1]
                final_list.append((sum_1, sum_2))
                distance = []
                next_time = times + int(fps + 1)
                if next_time > length:
                    next_time = length
                print (next_time)
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

#    print (final_list, len(final_list))
    return final_list[:55]
