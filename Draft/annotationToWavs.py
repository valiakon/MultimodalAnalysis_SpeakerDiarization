import glob
import os
import sys
import csv
import scipy.io.wavfile as wavfile
from pydub import AudioSegment
import shutil


def annotation2wavs(wavFile, csvFile):
    '''
        Break an audio stream to segments of interest, 
        defined by a csv file
        
        - wavFile:    path to input wavfile
        - csvFile:    path to csvFile of full annotated segments
        
        Input CSV file must be of the format <Tend>\t<Label>
    '''    
    audio = AudioSegment.from_wav(wavFile)

    start = 0
    with open(csvFile, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for i, row in enumerate(reader):
            t1 = int(row[0])
            end = t1 * 1000 #pydub works in millisec
            print ("split at [{}:{}] ms".format(start, end))
            audio_chunk = audio[start:end]
            audio_chunk.export("{}_{}_{}_{}.wav".format(wavFile, start, end, row[1]), format="wav")
            start = end  

def main(argv):
    if argv[1] == "-f":
        wavFile = argv[2]
        annotationFile = argv[3]
        annotation2wavs(wavFile, annotationFile)
    elif argv[1] == "-d":
        inputFolder = argv[2]
        types = ('*.txt', '*.csv')
        annotationFilesList = []
        for files in types:
            annotationFilesList.extend(glob.glob(os.path.join(inputFolder, files)))
        for anFile in annotationFilesList:
            wavFile = os.path.splitext(anFile)[0] + ".wav"
            if not os.path.isfile(wavFile):
                wavFile = os.path.splitext(anFile)[0] + ".mp3"
                if not os.path.isfile(wavFile):
                    print ("Audio file not found!")
                    return
            annotation2wavs(wavFile, anFile)


if __name__ == '__main__':
    # Used to extract a series of annotated WAV files based on (a) an audio file (mp3 or wav) and 
    # (b) an annotation file
    #
    # usage 1:
    # python annotateToWavs.py -f <audiofilepath> <annotationfilepath>
    # The <annotationfilepath> is actually a tab-seperated file where each line has the format <endTime>\t<classLabel>
    # The result of this process is a  series of WAV files with a file name <audiofilepath>_<startTime>_<endTime>_<classLabel>
    # 
    # usage 2:
    # python annotateToWavs.py -d <annotationfolderpath>
    # Same but searches all .txt and .csv annotation files. Audio files are supposed to be in the same path / filename with a WAV extension

    main(sys.argv)
