# !/usr/bin/env python3

import re
import sys
import time
import os
import pipes


def video_to_audio(fileName):
    file, file_extension = os.path.splitext(fileName)
    file = pipes.quote(file)
    os.system('ffmpeg -i ' + file + file_extension + ' ' + file + '.wav')
    # os.system('lame ' + file + '.wav' + ' ' + file + '.mp3')
    
def main():
    files = os.listdir("/Users/thanasiskaridis/videos_to_audio/")
    for f in files:
        print ("test\n" + f)
        if f[-3:] == "mp4":
           video_to_audio(f)

if __name__ == '__main__':
    main()
