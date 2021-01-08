from time import sleep
import numpy as np
import os

import json
import random
import torch

import pandas as pd
import training

SEED = 111222
#for idx, sub in enumerate([3,5,6,8,9,10,11,14,16,17,18,19,20,21,22,23,24,27,28,30]):
for idx, sub in enumerate([1,2,4,7,12,13,15,25,26,29]):
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    matsaved1 = np.zeros((45, 5))
    matsaved2 = np.zeros((45, 32))
    row_idx = 0
    
    for loo in range(1, 46):
        train = training.Train(sub, loo)
        train.run()
        del train
        
        i = open("rawresult.csv", 'r')
        temp = i.readlines()
        i.close()
        temp = temp[1].split()
        temp = np.array([float(idx) for idx in temp])
        matsaved1[row_idx] = temp
        del temp
        
        j = open("selected.csv", 'r')
        temp = j.readlines()
        j.close()
        if len(temp) < 2:
            temp = temp[0][1:-1].split()
        else:
            temp = (temp[0][1:-1] + temp[1][:-1]).split()
        temp = np.array([int(idx) for idx in temp])
        for i, x in enumerate(temp):
            matsaved2[row_idx][i] = x
        row_idx += 1
        del temp


    matsaved1 = pd.DataFrame(matsaved1)
    writer = pd.ExcelWriter('D://Dataset, Code//My Research//winter_vac//5000_0.01//pca_rawresult{}.xlsx'.format(sub), engine='openpyxl')
    matsaved1.to_excel(writer, index=False)
    writer.save()
    
    matsaved2 = pd.DataFrame(matsaved2)
    writer = pd.ExcelWriter('D://Dataset, Code//My Research//winter_vac//5000_0.01//pca_selected{}.xlsx'.format(sub), engine='openpyxl')
    matsaved2.to_excel(writer, index=False)
    writer.save()

    unique, counts = np.unique(matsaved2, return_counts=True)
    exec("unique{} = unique[1:]".format(sub))
    exec("counts{} = counts[1:]".format(sub))
    del unique, counts