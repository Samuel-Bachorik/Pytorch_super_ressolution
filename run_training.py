from process_dataset_2 import Process_dataset
from residual_model import Super_ress_model
import torch
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sobel_operator_model import Sobel_operator
from graphing_class import CreateGraph
if __name__ == '__main__':
    path =  ("C:/Users/Samuel/PycharmProjects/Super_ressolution/dataset3")
    loader = Process_dataset(path, in_ress=64, out_ress=512, aug_count=3)

    model = Super_ress_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # Time estimating variables
    epochminus, arrayloss, arrayepoch, lossforavg = 0, [], [], 0
    print(time.time())

    epoch_count = 200
    batch_size = 3
    batch_count = (loader.get_training_count() + batch_size) // batch_size
    print(batch_count, "Batch count")
    maxloss = 99999.0

    for epoch in range(epoch_count):

        epochminus += 1
        timestart = time.time()
        print("EPOCH - ", epoch)

        for batch in range(batch_count):
            x, y = loader.get_training_batch(batch_size)

            x = x.to(model.device)
            y = y.to(model.device)

            y_pred= model.forward(x)


            loss = ((y - y_pred) ** 2).mean()
            # Get loss number for graph
            lossforavg += float(loss.data.cpu().numpy())

            if epoch > 5 and maxloss > float(loss.data.cpu().numpy()):
                torch.save(model.state_dict(), './Model_lowest_loss')
                maxloss = float(loss.data.cpu().numpy())
                print("Model saved ")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        arrayepoch.append(epoch)
        arrayloss.append(lossforavg / batch_count)
        print(lossforavg / batch_count, "Epoch avg loss")
        lossforavg = 0

        timetoend = (epoch_count - epochminus) * (time.time() - timestart)
        dt_object = datetime.fromtimestamp(timetoend + time.time())
        print(dt_object, "time to end")

        # save model weights every 10th epoch
        if epoch % 10 == 0:
            PATH = './Model_epoch'
            torch.save(model.state_dict(), PATH)

        # Save final model
        PATH = './Model_final'
        torch.save(model.state_dict(), PATH)

        plt.plot(arrayepoch, arrayloss)
        plt.savefig('loss.png')
        plt.show()
