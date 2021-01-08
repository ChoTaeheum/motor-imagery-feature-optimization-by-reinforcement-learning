from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import pickle
import pandas as pd
import numpy as np
import hdf5storage
import random

#%% Random_number
rn = [11, 27, 8, 28, 17, 20, 15, 7, 1, 21, 41, 
      30, 12, 35, 23, 38, 44, 18, 5, 31, 13, 32, 
      34, 36, 29, 3, 37, 33, 39, 16, 43, 40, 9, 
      10, 2, 6, 26, 24, 25, 22, 42, 14, 0, 19, 4]   # 비복원 추출

#%% Data load, channel choice
for i in [2, 4, 5, 7, 8, 9, 10, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 29]:
    exec("data = np.array(hdf5storage.loadmat('Physionet_EEG_data_band_8to25Hz.mat')['subject_cell'][{}])".format(i)) # 서브젝트 no. - 1

    Ldata = len(data[0][0][0])   # 23, 그때그때 바뀜
    Rdata = len(data[1][0][0])   # 22
    
    data = np.transpose(np.concatenate((data[0], data[1]), 2), (2, 0, 1))
    data = data[:, [21, 23, 29, 31, 33, 35, 37, 38, 1, 3, 5, 39, 42, 40, 8, 10, 12, 41, 43,
                    44, 15, 17, 19, 45, 46, 48, 50, 52, 54, 60, 61, 62], :]    # 원하는 채널 선택 10-20
    
    label = np.concatenate((np.zeros(Ldata, ), np.ones(Rdata, )))
    
    #%% Dot product, Logarithm
    temp = np.zeros((45, 32))
    for idx, x in enumerate(data):
        temp[idx] = np.log(np.diag(np.dot(x, x.T)))  # 32개 채널 서로 내적해서 대각 요소
    
    a = []
    b = []
    for num in rn:  # 데이터 셔플링
        a.append(temp[num])
        b.append(label[num])
    
    pw_data = np.array(a)  # feature x 완성
    label = np.array(b).reshape(45, 1)
    
    del temp, a, b
    
    #%%
    # 설명력 90% 이상 principle component 개수
    expca = PCA(n_components=0.9)    # 90%의 설명력
    expca_data = expca.fit_transform(pw_data)
    exp90_pcnum = len(expca_data[0])
    del expca, expca_data
    
    pca = PCA()
    pca_data = pca.fit_transform(pw_data)
    dataset = np.concatenate((pca_data, label), axis=1)   # data + label
    
    #%% Data save, train=>30, evaluation=>15, 3-folded CV
#    dataset_train1 = pd.DataFrame(dataset[:30])
#    dataset_eval1 = pd.DataFrame(dataset[30:])
#    dataset_train1.to_pickle('.\sub{}\pca_tr1'.format(i+1))
#    dataset_eval1.to_pickle('.\sub{}\pca_ev1'.format(i+1))
#    
#    dataset_train2 = pd.DataFrame(np.concatenate((dataset[:15], dataset[30:45])))
#    dataset_eval2 = pd.DataFrame(dataset[15:30])
#    dataset_train2.to_pickle('.\sub{}\pca_tr2'.format(i+1))
#    dataset_eval2.to_pickle('.\sub{}\pca_ev2'.format(i+1))
#    
#    dataset_train3 = pd.DataFrame(dataset[15:])
#    dataset_eval3 = pd.DataFrame(dataset[:15])
#    dataset_train3.to_pickle('.\sub{}\pca_tr3'.format(i+1))
#    dataset_eval3.to_pickle('.\sub{}\pca_ev3'.format(i+1))

    # data와 label을 따로!
    pd.DataFrame(pca_data).to_pickle('.\sub{}\pca_data'.format(i+1))
    pd.DataFrame(label).to_pickle('.\sub{}\pca_label'.format(i+1))
    
    # data + label
#    pd.DataFrame(dataset).to_pickle('.\sub{}\pca_dataset'.format(i+1))   # for reinforcement learning
    
    #%% Leave-one out
    for j in range(0, 45):     # 1,2,3,4,5, ..., 45
        exec("loo_train{} = pd.DataFrame(np.concatenate((dataset[:{}], dataset[{}:])))".format(j+1, j, j+1))
        exec("loo_eval{} = pd.DataFrame(dataset[{}]).T".format(j+1, j))
        exec("loo_train{}.to_pickle('.\sub{}\pca_loo_tr{}')".format(j+1, i+1, j+1))
        exec("loo_eval{}.to_pickle('.\sub{}\pca_loo_ev{}')".format(j+1, i+1, j+1))
