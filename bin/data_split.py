#!/usr/bin/env python3


from PIL import Image

import glob

images = []

imagesList = {}
imagesList = glob.glob('*.png')
imagesList.sort()

imagesListSize = len(imagesList)
trainDataSize = round(imagesListSize * 0.7)

f1 = open("train_data.csv","w+")
f2 = open("test_data.csv","w+")
#f3 = open("validation_data.csv","w+")

counter=0
while (counter < trainDataSize):
    f1.write("maksssksksss" + str(counter) + ".png\n")
    counter+=1

counter=int(trainDataSize)
while (counter < imagesListSize):
    f2.write("maksssksksss" + str(counter) + ".png\n")
    counter+=1

f1.close()
f2.close()
#f3.close()
    

   

        
        

        


