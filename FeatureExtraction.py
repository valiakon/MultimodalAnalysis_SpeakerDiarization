from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from pydub import AudioSegment
import matplotlib.pyplot as plt



def StereoToMono(wavFilePath):
    sound = AudioSegment.from_wav(wavFilePath)
    sound = sound.set_channels(1)
    outPath = "/home/valia/Desktop/clintonMono.wav"
    sound.export(outPath, format="wav")
    return outPath


def ExtractFeatures(newPath):
    [Fs, x] = audioBasicIO.readAudioFile(newPath)
    F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
    print F

def main():
    wavFilePath = "/home/valia/Downloads/clinton1.wav"
    print wavFilePath
    newPath = StereoToMono(wavFilePath)
    ExtractFeatures(newPath)

if __name__== "__main__":
  main()


