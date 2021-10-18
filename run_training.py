from process_dataset_2 import Process_dataset
from Model import Super_ress_model
import numpy
from PIL import Image
import torch
import os
import time
from datetime import datetime


if __name__ == '__main__':
    path =  ("C:/Users/Samuel/PycharmProjects/Super_ressolution/dataset")
    loader = Process_dataset(path,128,128,3)

    model = Super_ress_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # Time estimating variables
    epochminus, arrayloss, arrayepoch, lossforavg = 0, [], [], 0
    print(time.time())

    epoch_count = 100
    batch_size = 20
    batch_count = (loader.get_training_count() + batch_size) // batch_size
    print(batch_count, "Batch count")

    for epoch in range(epoch_count):

        epochminus += 1
        timestart = time.time()
        print("EPOCH - ", epoch)

        for i in range(batch_count):
            x, y = loader.get_training_batch(batch_size)

            x = x.to(model.device)
            y = y.to(model.device)

            im = x[0]
            # im = numpy.swapaxes(im,1,0)
            im_y = y[0]
            im = im.detach().to("cpu").numpy()
            im_y = im_y.detach().to("cpu").numpy()

            im = numpy.moveaxis(im, 0, 2)
            im_y = numpy.moveaxis(im_y, 0, 2)

            im = (im * 255).astype(numpy.uint8)
            im_y = (im_y * 255).astype(numpy.uint8)
            imgg = Image.fromarray(im)
            imgg2 = Image.fromarray(im_y)
            # imgg = imgg.rotate(-90, PIL.Image.NEAREST, expand=1)
            #imgg.show()
            #imgg2.show()

            y_pred= model.forward(x)

            loss = ((y - y_pred) ** 2).mean()
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        timetoend = (epoch_count - epochminus) * (time.time() - timestart)
        dt_object = datetime.fromtimestamp(timetoend + time.time())
        print(dt_object, "time to end")



