import numpy as np
import pickle
import pandas as pd
import numpy as np
import hdf5storage
import random
from CSP import CSP
from sklearn.model_selection import KFold

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

#%%
rns = [11, 27, 8, 28, 17, 20, 15, 7, 1, 21, 41, 
      30, 12, 35, 23, 38, 44, 18, 5, 31, 13, 32, 
      34, 36, 29, 3, 37, 33, 39, 16, 43, 40, 9, 
      10, 2, 6, 26, 24, 25, 22, 42, 14, 0, 19, 4]   # 비복원 추출

for i in range(30):
    exec("data = np.array(hdf5storage.loadmat('Physionet_EEG_data_band_8to25Hz.mat')['subject_cell'][{}])".format(i)) # 서브젝트 no. - 1
    
    Ldata_num = len(data[0][0][0])   # 23, 그때그때 바뀜
    Rdata_num = len(data[1][0][0])   # 22
    
    data = np.transpose(np.concatenate((data[0], data[1]), 2), (2, 0, 1))
    data = data[:, [21, 23, 29, 31, 33, 35, 37, 38, 1, 3, 5, 39, 42, 40, 8, 10, 12, 41, 43,
                    44, 15, 17, 19, 45, 46, 48, 50, 52, 54, 60, 61, 62], :]    # 원하는 채널 선택 10-20
    
    label = np.concatenate((np.zeros(Ldata_num, ), np.ones(Rdata_num, )))
    
    kf = KFold(n_splits=45)
    for train_idx, test_idx in kf.split(data):
        tr_data = data[train_idx]        
        tr_label = label[train_idx]
        ts_data = data[test_idx]
        ts_label = data[test_idx]
        
        
        
    
    #%%
    w = CSP(Ldata, Rdata)     # CSP filter
    
    # Z = w*x
    Z = np.zeros((45, 32, 640))
    temp = np.zeros((45, 32))
    for idx, x in enumerate(data):
        Z[idx] = w[0] @ x     # (32, 32) * (32, 640) => (32, 640)
        temp[idx] = np.log(np.var(Z[idx], axis=1)/np.sum(np.var(Z[idx], axis=1)))
        
    a = []
    b = []
    for rn in rns:  # 데이터 셔플링
        a.append(temp[rn])
        b.append(label[rn])
        
    csp_data = np.array(a)  # feature x 완성
    label = np.array(b).reshape(45, 1)
    dataset = np.concatenate((csp_data, label), axis=1)   # data + label
    del temp, a, b
    
    #%% Data save, 3-folded CV
#    dataset_train1 = pd.DataFrame(dataset[:30])
#    dataset_eval1 = pd.DataFrame(dataset[30:])
#    dataset_train1.to_pickle('.\sub{}\csp_tr1'.format(i+1))
#    dataset_eval1.to_pickle('.\sub{}\csp_ev1'.format(i+1))
#    
#    dataset_train2 = pd.DataFrame(np.concatenate((dataset[:15], dataset[30:45])))
#    dataset_eval2 = pd.DataFrame(dataset[15:30])
#    dataset_train2.to_pickle('.\sub{}\csp_tr2'.format(i+1))
#    dataset_eval2.to_pickle('.\sub{}\csp_ev2'.format(i+1))
#    
#    dataset_train3 = pd.DataFrame(dataset[15:])
#    dataset_eval3 = pd.DataFrame(dataset[:15])
#    dataset_train3.to_pickle('.\sub{}\csp_tr3'.format(i+1))
#    dataset_eval3.to_pickle('.\sub{}\csp_ev3'.format(i+1))
#    
#    # data와 label을 따로!  
#    pd.DataFrame(csp_data).to_pickle('.\sub{}\csp_data'.format(i+1))
#    pd.DataFrame(label).to_pickle('.\sub{}\csp_label'.format(i+1))
#    
#    # data + label
#    pd.DataFrame(dataset).to_pickle('.\sub{}\csp_dataset'.format(i+1))
#    
    #%% Leave-one out    
    for j in range(0, 45):     # 1,2,3,4,5, ..., 45
        exec("loo_train{} = pd.DataFrame(np.concatenate((dataset[:{}], dataset[{}:])))".format(j+1, j, j+1))
        exec("loo_eval{} = pd.DataFrame(dataset[{}]).T".format(j+1, j))
        exec("loo_train{}.to_pickle('.\sub{}\csp_loo_tr{}')".format(j+1, i+1, j+1))
        exec("loo_eval{}.to_pickle('.\sub{}\csp_loo_ev{}')".format(j+1, i+1, j+1))
