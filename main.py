import torch
import PIL
from PIL import Image, ImageOps
import numpy
import math
import os
x = Image.open("20210708_101351.jpg")
x = ImageOps.exif_transpose(x)

"""
#Resize
if x.size[0]> x.size[1]:
    z = (x.size[0] / x.size[1]) * 768
    x = x.resize((math.floor(z), 768))
else:
    z = (x.size[1] / x.size[0]) * 768
    x = x.resize((256, math.floor(z)))

x = x.crop((math.floor((x.size[0]-768)/2),0 ,math.floor((x.size[0]-768)/2)+768, 768))
print(x.size)
x.show()
#############################################
# Flip
x = x.transpose(PIL.Image.FLIP_LEFT_RIGHT)
x = x.transpose(PIL.Image.FLIP_TOP_BOTTOM)

#rotate
x = x.rotate(25)
"""
path = ("C:/Users/Samuel/PycharmProjects/Super_ressolution/dataset2")
f = os.listdir(path)
print(len(f))
class ImagesLoader:
    def __init__(self, path,height,width):
        self.path   = path
        self.height = height
        self.width  = width
        self.channels = 3

        f = os.listdir(path)
        self.dataset_count = len(f)
        self.images = numpy.zeros((self.dataset_count, self.channels, self.height, self.width), dtype=numpy.uint8)
        self.labels = numpy.zeros((self.dataset_count, self.channels, 384, 384), dtype=numpy.uint8)

        print(self.images.shape)
        #self.load_images()

    def load_images(self):
        for i in range(len(f)):
            y = Image.open(os.path.join(path, f[i]))
            #x = ImageOps.exif_transpose(x)

            if y.size[0] > y.size[1]:
                new_size = (y.size[0] / y.size[1]) * 384
                y = y.resize((math.floor(new_size), 384))
            else:
                new_size = (y.size[1] / y.size[0]) * 384
                y = y.resize((384, math.floor(new_size)))

            y = y.crop((math.floor((y.size[0] - 384) / 2), 0, math.floor((y.size[0] - 384) / 2) + 384, 384))

            #y.show()
            y_np = numpy.array(y)
            y_np = numpy.moveaxis(y_np,-1,0)
            self.labels[i] = y_np

            x = y.resize((128,128))
            #x.show()
            x_np = numpy.array(x)
            x_np = numpy.moveaxis(x_np,-1,0)
            self.images[i] = x_np

        return self.images,self.labels