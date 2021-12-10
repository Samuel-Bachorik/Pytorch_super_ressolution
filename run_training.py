from process_dataset_2 import Process_dataset
from small_residual_model import Super_ress_model
import numpy
from PIL import Image
import torch
import os
import time
from datetime import datetime
import math
import matplotlib.pyplot as plt
from sobel_operator_model import Sobel_operator
from graphing_class import CreateGraph

if __name__ == '__main__':
    path =  ("C:/Users/Samuel/PycharmProjects/Super_ressolution/dataset2")
    loader = Process_dataset(path, in_ress=64, out_ress=512, aug_count=3)

    sobel_operator = Sobel_operator()

    model = Super_ress_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # Time estimating variables
    epochminus, arrayloss, arrayepoch, lossforavg = 0, [], [], 0
    print(time.time())

    epoch_count = 100
    batch_size = 7
    PSNR_count = 0
    learning_rate = 0.0001
    batch_count = (loader.get_training_count() + batch_size) // batch_size
    print(batch_count, "Batch count")
    maxloss = 99999.0

    loss_graph = CreateGraph(batch_count,"LOSS")
    edges_graph = CreateGraph(batch_count, "EDGES")
    MSE_graph = CreateGraph(batch_count, "MSE")

    for epoch in range(epoch_count):
        epochminus += 1
        timestart = time.time()
        print("EPOCH - ", epoch)

        if epoch > 0:
            learning_rate = 0.00001

        print(learning_rate,"learning rate ")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for batch in range(batch_count):
            x, y = loader.get_training_batch(batch_size)

            x = x.to(model.device)
            y = y.to(model.device)

            y_pred= model.forward(x)
            edges = sobel_operator.find_edges(y_pred)

            loss_mse = ((y - y_pred) ** 2).mean()
            loss_edges = -0.001 * (edges ** 2).mean()
            loss = loss_mse + loss_edges


            print(loss_mse,loss_edges)

            loss_graph.num_for_avg += float(loss.data.cpu().numpy())
            MSE_graph.num_for_avg += float(loss_mse.data.cpu().numpy())
            edges_graph.num_for_avg += float(loss_edges.data.cpu().numpy())

            if epoch > 5 and maxloss > float(loss.data.cpu().numpy()):
                torch.save(model.state_dict(), './Model_sobel1_loss')
                maxloss = float(loss.data.cpu().numpy())
                print("Model saved ")


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            PSNR = 10*torch.log10(1/((y - y_pred)**2).mean())
            PSNR_count+= PSNR.detach().to("cpu").numpy()

        print("PSNR - ",PSNR_count/epoch_count)
        PSNR_count = 0

        loss_graph.count(epoch)
        edges_graph.count(epoch)
        MSE_graph.count(epoch)

        timetoend = (epoch_count - epochminus) * (time.time() - timestart)
        dt_object = datetime.fromtimestamp(timetoend + time.time())
        print(dt_object, "time to end")

        # save model weights every 5th epoch
        if epoch % 5 == 0:
            PATH = './Model_sobel_1_epoch.pth'
            torch.save(model.state_dict(), PATH)
