import torch
import PIL
from PIL import Image, ImageOps
import numpy
import math
import os
from main import ImagesLoader
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
class Process_dataset:
    def __init__(self,path, height,width,aug_count):
        self.height = height
        self.width = width
        self.path = path
        self.aug_count = aug_count
        self.training_images = []
        self.training_labels = []
        self.path =  ("C:/Users/samue/PycharmProjects/reinforcement_learning_env/dataset2")
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


    def get_training_batch(self,batch_size):
        return self.get_batch(self.training_images,32)

    def get_batch(self,images,batch_size):
        result_x = torch.zeros((batch_size, 3, 128, 128)).float()
        result_y = torch.zeros((batch_size, 3, 384, 384)).float()

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = [None] * batch_size
            for x in range(batch_size):
                results[x] = executor.submit(self.process_batch,self.training_images,self.training_labels)

            counter = 0
            for f in concurrent.futures.as_completed(results):
                result_x[counter], result_y[counter] = f.result()[0], f.result()[1]
                counter += 1

        return result_x, result_y



    def process_batch(self,images,labels):
        group_idx = numpy.random.randint(len(self.training_images))
        image_idx = numpy.random.randint(len(self.training_images[group_idx]))


        image_np = numpy.array(images[group_idx][image_idx]) / 256.0
        label_np = numpy.array(labels[group_idx][image_idx])/256.0

        # if self._rnd(0, 1) > 0.1:

        image_np, mask_np = self._augmentation_flip(image_np, label_np)


        result_x = torch.from_numpy(numpy.flip(image_np,axis=0).copy())
        result_y = torch.from_numpy(numpy.flip(mask_np,axis=0).copy())

        return result_x, result_y


    def _augmentation_flip(self, image_np,label_np):
        if self._rnd(0, 1) < 0.5:
            aug_img = numpy.flip(image_np, 1)
            aug_label = numpy.flip(label_np, 1)
        else:
            aug_img = numpy.flip(image_np, 2)
            aug_label = numpy.flip(label_np, 2)

        return aug_img,aug_label



    def _rnd(self, min_value, max_value):
        return (max_value - min_value) * numpy.random.rand() + min_value

path  =("C:/Users/samue/PycharmProjects/reinforcement_learning_env/dataset2")
Process_dataset(path,128,128,10)
