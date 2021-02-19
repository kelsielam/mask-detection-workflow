#!/usr/bin/env python3


from PIL import Image
import glob
from numpy import asarray


def norm_image(img):
    """
    Normalize PIL image

    """
    pixels = asarray(img)
    pixels = pixels.astype('float32')
    pixels /= 255.0
    img_nrm = Image.fromarray(pixels,'RGB')

    return img_nrm



def main():
	imagesList = glob.glob('*.png')

	for item in imagesList:
		img = Image.open(item)
		img_nrm = norm_image(img)
		img_nrm = img_nrm.save("preprocessed_"+item)

        


if __name__ == '__main__':
    main()