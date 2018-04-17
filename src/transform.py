import os
from scipy import ndimage
from scipy import misc
import numpy as np

files = []

with open('face_bbx.txt') as file:

    currentFile = None
    currentImage = None

    for line in file:
        line = line.replace('\n', '')
        if (len(line) < 10):
            continue
        if (line[-3:] == 'jpg'):
            if (currentFile):
                currentFile.close()
            currentFile = open("./labels/" + line[0:-4] + '.txt', "w+")
            currentImage = misc.imread("./images/" + line).shape
            files.append(line)
        else:
            d = line.split(' ')[0:4]
            currentFile.write('0 ' +
                              str(int(d[0]) / currentImage[1]) +
                              ' ' +
                              str(int(d[1]) / currentImage[0]) +
                              ' ' +
                              str(int(d[2]) / currentImage[1]) +
                              ' ' +
                              str(int(d[3]) / currentImage[0]) +
                              '\n')

    currentFile.close()
