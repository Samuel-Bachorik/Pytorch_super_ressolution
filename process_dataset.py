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
        self.path =  ("C:/Users/Samuel/PycharmProjects/Super_ressolution/dataset2")
        Loader = ImagesLoader(self.path, 128,128)
        images,labels = Loader.load_images()
        from PIL import Image


        #(6, 3, 128, 128)
        print(images.shape,"KOKOKOKOKO")
        #(6, 3, 128, 128) KOKOKOKOKO

        self.training_images.append(images)
        self.training_labels.append(labels)


        images,labels = self.process_augumentation(self.training_images,self.training_labels,2,16)
        self.training_images.append(images)
        self.training_labels.append(labels)


    def process_augumentation(self,images,labels,aug_count,batch_count):
        count = len(images[0])
        total_count = count * aug_count

        aug_images = numpy.zeros((total_count, 3, 128, 128), dtype=numpy.uint8)
        aug_labels = numpy.zeros((total_count, 3, 384, 384), dtype=numpy.uint8)
        counter = 0
        for i in range(aug_count):
            for q in range(count):
                aug_img = images[0][q]
                print(aug_img.shape,"AUG IMG")
                aug_label = labels[0][q]

                brightness = self._rnd(-0.25, 0.25)
                contrast = self._rnd(0.5, 1.5)
                noise = 0.05 * (2.0 * numpy.random.rand(3, 384, 384) - 1.0)
                noise_low = numpy.swapaxes(noise,0,2)
                noise_low = cv2.resize(noise_low, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                noise_low = numpy.swapaxes(noise_low, 0, 2)

                img_result = aug_img
                img_result = img_result + brightness
                img_result = 0.5 + contrast * (img_result - 0.5)
                img_result = img_result + noise_low
                img_result = numpy.clip(img_result, 0.0, 1.0)  # možno

                aug_label = aug_label
                label_result = aug_label + brightness
                label_result = 0.5 + contrast * (label_result - 0.5)
                label_result = label_result + noise
                #label_result = numpy.clip(label_result, 0.0, 1.0)  # možno


                im = label_result
                # im = im.detach().to("cpu").numpy()
                im = numpy.moveaxis(im, 0, 2)
                im = im.astype(numpy.uint8)
                imgg = Image.fromarray(im)
                #imgg.show()


                aug_images[counter] = img_result
                aug_labels[counter] = label_result
                counter+=1

        return aug_images,aug_labels


    def get_training_batch(self,batch_size):

        """"x = self.training_images[1][0][0]
        im = x
        # im = im.detach().to("cpu").numpy()
        im = numpy.moveaxis(im, 0, 2)
        #im = im.astype(numpy.uint8)
        imgg = Image.fromarray(im)
        #imgg.show()"""

        return self.get_batch(self.training_images,32)

    def get_batch(self,images,batch_size):
        result_x = torch.zeros((batch_size, 3, 128, 128)).float()
        result_y = torch.zeros((batch_size, 3, 384, 384)).float()

        with ThreadPoolExecutor(max_workers=1) as executor:
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

        image_np = numpy.array(images[group_idx][image_idx])/255.0
        label_np = numpy.array(labels[group_idx][image_idx])
        print(numpy.min(image_np), numpy.max(image_np ),"KOKOT",numpy.min(label_np), numpy.max(label_np ))
        # if self._rnd(0, 1) > 0.1:

        image_np, mask_np = self._augmentation_flip(image_np, label_np)

        result_x = torch.from_numpy(image_np)
        result_y = torch.from_numpy(mask_np)

        return result_x, result_y


    def _augmentation_flip(self, image_np,label_np):
        #if self._rnd(0, 1) < 0.5:
          #  aug_img = numpy.flip(image_np, 1)
         #   aug_label = numpy.flip(label_np, 1)
        #else:
            #aug_img = numpy.flip(image_np, 2)
            #aug_label = numpy.flip(label_np, 2)

        aug_img = numpy.flip(image_np, 1)
        aug_label = numpy.flip(label_np, 1)

        return aug_img,aug_label



    def _rnd(self, min_value, max_value):
        return (max_value - min_value) * numpy.random.rand() + min_value

