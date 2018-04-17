import random
import os


val = open("./val.txt", "w+")
train = open("./train.txt", "w+")

path = os.path.dirname(os.path.realpath(__file__))

with open('face_bbx.txt') as file:
    for line in file:
        print(line[-4:-1], line[-4:-1] == 'jpg')
        if (line[-4:-1] == 'jpg'):
            if (random.random() < 7/16):
                val.write(str(path) + '/images/' + line)
            else:
                train.write(str(path) + '/images/' + line)

val.close()
train.close()
