import torch
import PIL
from PIL import Image, ImageOps
import numpy
import math
import os
import cv2
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
        self.path = ("C:/Users/Samuel/PycharmProjects/Super_ressolution/dataset2")
        Loader = ImagesLoader(self.path, 128, 128)
        images, labels = Loader.load_images()

        self.training_images.append(images)
        self.training_labels.append(labels)
        print("HOVNO")

    def get_training_batch(self,batch_size):
        return self.get_batch(self.training_images,self.training_labels,32)


    def get_batch(self,images,labels,batch_size):
        result_x = torch.zeros((batch_size, 3, 128, 128)).float()
        result_y = torch.zeros((batch_size, 3, 384, 384)).float()

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = [None] * batch_size
            for x in range(batch_size):
                results[x] = executor.submit(self.process_batch,images,labels)

            counter = 0
            for f in concurrent.futures.as_completed(results):
                result_x[counter], result_y[counter] = f.result()[0], f.result()[1]
                counter += 1

        return result_x, result_y


    def process_batch(self,images,labels):
        group_idx = numpy.random.randint(len(self.training_images))
        image_idx = numpy.random.randint(len(self.training_images[group_idx]))

        image_np = numpy.array(images[group_idx][image_idx])
        label_np = numpy.array(labels[group_idx][image_idx])
        #print(numpy.min(image_np), numpy.max(image_np ),"KOKOT",numpy.min(label_np), numpy.max(label_np ))
        # if self._rnd(0, 1) > 0.1:
        image_np, mask_np = self._augmentation_flip(image_np, label_np)
        #image_np, mask_np = self._augmentation_noise(image_np, mask_np)

        result_x = torch.from_numpy(image_np)
        result_y = torch.from_numpy(mask_np)

        return result_x, result_y

    def _augmentation_flip(self, image_np,label_np):
        if self._rnd(0, 1) < 0.5:
            aug_img = numpy.flip(image_np, 1)
            aug_label = numpy.flip(label_np, 1)
        else:
            aug_img = numpy.flip(image_np, 2)
            aug_label = numpy.flip(label_np, 2)

        return aug_img.copy(), aug_label.copy()

    def _augmentation_noise(self, image_np,label_np):
        brightness = self._rnd(-0.25, 0.25)
        contrast = self._rnd(0.5, 1.5)
        noise = 0.05 * (2.0 * numpy.random.rand(3, 384, 384) - 1.0)

        noise_low = numpy.swapaxes(noise, 0, 2)
        noise_low = cv2.resize(noise_low, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        noise_low = numpy.swapaxes(noise_low, 0, 2)

        img_result = image_np + brightness
        img_result = 0.5 + contrast * (img_result - 0.5)
        img_result = img_result + noise_low

        label_result = label_np + brightness
        label_result = 0.5 + contrast * (label_result - 0.5)
        label_result = label_result + noise

        return numpy.clip(img_result, 0.0, 1.0),numpy.clip(label_result, 0.0, 1.0)
        #return img_result,label_result


    def _rnd(self, min_value, max_value):
        return (max_value - min_value) * numpy.random.rand() + min_value