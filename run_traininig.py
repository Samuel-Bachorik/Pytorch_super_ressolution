from process_dataset import Process_dataset


path =  ("C:/Users/samue/PycharmProjects/reinforcement_learning_env/dataset2")
loader = Process_dataset(path,128,128,3)

x,y = loader.get_training_batch(32)

print(x.shape)
print(y.shape)
import numpy

from PIL import Image

z = numpy.array(y)

z = z*256.0
z = z[3]
z = z.astype(numpy.uint8)
print(z.shape)
z = numpy.swapaxes(z,0,2)

image = Image.fromarray(z)
image.show()
