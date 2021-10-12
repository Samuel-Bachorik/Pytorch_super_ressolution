import torch
import PIL
from PIL import Image, ImageOps
import numpy
import math
import os
from main import ImagesLoader
class Process_dataset:
    def __init__(self,path, height,width,aug_count):
        self.height = height
        self.width = width
        self.path = path
        self.aug_count = aug_count
        self.training_images = []
        self.training_labels = []
        self.path =  ("C:/Users/Samuel/PycharmProjects/Super_ressolution/dataset2")
        Loader = ImagesLoader(self.path, 128,128)
        self.images,self.labels = Loader.load_images()
        #(6, 3, 128, 128)
        #print(self.images.shape)

        self.training_images.append(self.images)
        self.training_labels.append(self.labels)
        self.process_augumentation(self.training_images,self.training_labels,2,16)


    def process_augumentation(self,images,labels,aug_count,batch_count):
        for i in range(aug_count):
            for q in range(self.images.shape[0]):
                aug_images = numpy.zeros((self.images.shape[0], 3, 128, 128), dtype=numpy.uint8)
                aug_labels = numpy.zeros((self.labels.shape[0], 3, 384, 384), dtype=numpy.uint8)

                aug_img = images[i][q]
                aug_label = labels[i][q]
                brightness = self._rnd(-0.25, 0.25)
                contrast = self._rnd(0.5, 1.5)

                img_result = aug_img + brightness
                img_result = 0.5 + contrast * (img_result - 0.5)
                img_result = numpy.clip(img_result, 0.0, 1.0)  # možno

                label_result = aug_label + brightness
                label_result = 0.5 + contrast * (label_result - 0.5)
                label_result = numpy.clip(label_result, 0.0, 1.0)  # možno

                aug_images[q] = img_result
                aug_labels[q] = label_result

            self.training_images.append(aug_images)
            self.training_labels.append(aug_labels)


    def get_batch(self,images):

        group_idx = numpy.random.randint(len(images))
        image_idx = numpy.random.randint(len(images[group_idx]))


    def process_batch(self):
        pass

    def _augmentation_flip(self, image_np,label_np):
        if self._rnd(0, 1) < 0.5:
            aug_img = numpy.flip(image_np, 1)
            aug_label = numpy.flip(label_np, 1)
        else:
            aug_img = numpy.flip(image_np, 2)
            aug_label = numpy.flip(label_np, 2)



    def _rnd(self, min_value, max_value):
        return (max_value - min_value) * numpy.random.rand() + min_value

path  =("C:/Users/Samuel/PycharmProjects/Super_ressolution/dataset2")
Process_dataset(path,128,128,10)