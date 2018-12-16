# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import os

def findVideos():
    for root, dirs, files in os.walk("/home/valia/Desktop/all_videos"):
        for file in files:
            if file.endswith(".mp4"):
                print "fil"#(os.path.join(root, file))
    return (os.path.join(root, file))
                
                
                
                

def main():
  pathLast = findVideos()
  print pathLast
  
if __name__== "__main__":
  main()