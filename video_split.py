#!/usr/bin/env python

import subprocess
import math
from optparse import OptionParser

def splitByTime(filename, duration, vcodec="copy", acodec="copy", **kwargs):
    videoLength = int(float(subprocess.check_output(("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                                    "default=noprint_wrappers=1:nokey=1", filename)).strip()))
    files = int(math.ceil(videoLength / float(duration)))

    for n in range(0, files):
        args = []
        if n == 0:
            split_start = 0
        else:mu
            split_start = duration * n
        args += ["-ss", str(split_start), "-t", str(duration),
                 ".".join(filename.split(".")[:-1]) + str(n + 1) + "." + filename.split(".")[-1]]
        subprocess.check_output(["ffmpeg", "-i", filename, "-vcodec", vcodec, "-acodec", acodec] + args)


def main():
    parser = OptionParser()
    parser.add_option("-f", dest="filename", type="string")
    parser.add_option("-s", dest="duration", type="int")
    (options, args) = parser.parse_args()

    splitByTime(**(options.__dict__))


if __name__ == '__main__':
    main()

# you have to run this .py like:
# python ffmpeg-split.py -f ellendegeneres-jimmyfallon.mp4 -s 60