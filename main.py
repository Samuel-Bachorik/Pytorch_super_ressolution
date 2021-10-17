import torch
import PIL
from PIL import Image, ImageOps
import numpy
import math
import os

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


        #self.load_images()

    def load_images(self):
        f = os.listdir(self.path)
        print(len(f))
        for i in range(len(f)):
            y = Image.open(os.path.join(self.path, f[i]))
            #x = ImageOps.exif_transpose(x)

            if y.size[0] > y.size[1]:
                new_size = (y.size[0] / y.size[1]) * 384
                y = y.resize((math.floor(new_size), 384))
            else:
                new_size = (y.size[1] / y.size[0]) * 384
                y = y.resize((384, math.floor(new_size)))

            y = y.crop((math.floor((y.size[0] - 384) / 2), 0, math.floor((y.size[0] - 384) / 2) + 384, 384))

            y_np = numpy.array(y)
            y_np = numpy.moveaxis(y_np,-1,0)
            self.labels[i] = y_np

            x = y.resize((128,128))
            #x.show()
            x_np = numpy.array(x)
            x_np = numpy.moveaxis(x_np,-1,0)


            self.images[i] = x_np


        return self.images,self.labels
