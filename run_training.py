from process_dataset_2 import Process_dataset
from Model import Super_ress_model
import numpy
from PIL import Image
import torch
import os
import time
from datetime import datetime

path =  ("C:/Users/Samuel/PycharmProjects/Super_ressolution/dataset2")
loader = Process_dataset(path,128,128,3)

model = Super_ress_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for i in range(100):
    x, y = loader.get_training_batch(32)
    x = x.to(model.device)
    y = y.to(model.device)

    y_pred= model.forward(x)

    loss = ((y-y_pred)**2).mean()
    print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




"""im = y[9]
im = im.detach().to("cpu").numpy()
im = numpy.moveaxis(im, 0, 2)
img = (im*255).astype(numpy.uint8)
imgg = Image.fromarray(img)
imgg.show()"""

