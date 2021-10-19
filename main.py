import torch
import PIL
from PIL import Image, ImageOps
import numpy
import math
import os

class ImagesLoader:
    def __init__(self, path,in_ress,out_ress):
        self.path   = path
        self.in_ress = in_ress
        self.out_ress  = out_ress
        self.channels = 3

        f = os.listdir(path)
        self.dataset_count = len(f)
        self.images = numpy.zeros((self.dataset_count, self.channels, self.in_ress, self.in_ress), dtype=numpy.uint8)
        self.labels = numpy.zeros((self.dataset_count, self.channels, self.out_ress, self.out_ress), dtype=numpy.uint8)


    def load_images(self):
        f = os.listdir(self.path)
        print(len(f))
        for i in range(len(f)):
            y = Image.open(os.path.join(self.path, f[i]))
            #x = ImageOps.exif_transpose(x)

            if y.size[0] > y.size[1]:
                new_size = (y.size[0] / y.size[1]) * self.out_ress
                y = y.resize((math.floor(new_size), self.out_ress))
            else:
                new_size = (y.size[1] / y.size[0]) * self.out_ress
                y = y.resize((self.out_ress, math.floor(new_size)))

            y = y.crop((math.floor((y.size[0] - self.out_ress) / 2), 0, math.floor((y.size[0] - self.out_ress) / 2) + self.out_ress, self.out_ress))

            y_np = numpy.array(y)
            y_np = numpy.moveaxis(y_np,-1,0)
            self.labels[i] = y_np

            x = y.resize((self.in_ress,self.in_ress))

            x_np = numpy.array(x)
            x_np = numpy.moveaxis(x_np,-1,0)

            self.images[i] = x_np


        return self.images,self.labels
