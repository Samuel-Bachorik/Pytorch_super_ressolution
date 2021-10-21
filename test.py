import numpy
import torch
"""
x = numpy.zeros((12,3,48,48))
print(x)

array = []

array.append(x)
print(array)

print(array[0].shape)
print(array[0][0].shape)
print(array[0][0][0].shape)
(12, 3, 48, 48)
(3, 48, 48)
(48, 48)
import cv2
img = cv2.imread('20210708_101351.jpg')
print(img.shape

"""


from PIL import Image
import numpy
import torch
from Model import Super_ress_model
import PIL

with torch.no_grad():
    model = Super_ress_model()
    model.eval()
    im = Image.open("toyota low res.jpg")
    im_low = im



    im = numpy.array(im)


    im = numpy.expand_dims(im,0)
    im = numpy.swapaxes(im,1,3)/255.0
    print(im.shape)
    im = torch.from_numpy(im).float()
    im =  im.to(model.device)
    im = im.cuda()
    PATH = "Model_lowest_loss_2.pth"
    model.load_state_dict(torch.load(PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pred = model.forward(im)

    k = 1.0 / (torch.max(pred) - torch.min(pred))
    q = 1 - k * torch.max(pred)

    y_norm = k*pred + q
    print(torch.max(y_norm))
    print(torch.min(y_norm))
    #pred = torch.max(pred, 0)


    im = y_norm[0]
    # im = numpy.swapaxes(im,1,0)
    im = im.detach().to("cpu").numpy()
    im = numpy.moveaxis(im, 0, 2)
    im = (im * 255.0).astype(numpy.uint8)

    imgg = Image.fromarray(im)

    imgg = imgg.rotate(-90, PIL.Image.NEAREST, expand=1)
    imgg = imgg.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    im_low = im_low.resize((512, 512), Image.NEAREST)
    im_low.save("toyota LOWWWWW 512.jpg")
    #imgg.show()
    imgg.save("toyota high ress.jpg")
