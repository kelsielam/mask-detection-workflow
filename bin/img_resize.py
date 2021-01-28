#!/usr/bin/env python3


from PIL import Image
import glob


imagesList = glob.glob('*.png')

for item in imagesList:
    img = Image.open(item)
    newsize = (224, 224)
    img = img.resize(newsize)
    img = img.save("resized_"+item")
    
   

        
        

        


