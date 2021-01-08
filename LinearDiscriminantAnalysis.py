from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import pickle

subs_acc = np.zeros((10,32))
#subs_acc = np.zeros((10,16))
#rank = np.array(pd.read_excel('rank.xlsx'))

for idx, sub in enumerate([1, 2, 4, 7, 12, 13, 15, 25, 26, 29]):
#    feat_ord = rank[:,idx] - 1  # each subject에 해당하는 column 반환
    
    for m in range(32):
#    for m in range(16):
        
        exec("data = np.array(pd.read_pickle('./sub{}/pca_data'))".format(sub))
        exec("label = np.array(pd.read_pickle('./sub{}/pca_label'))".format(sub))
        
#        data = data[:,feat_ord[:m+1]]    # new feature order
        
        data = data[:,:m+1]
        
        # for csp
#        fr_m = [e for e in range(m+1)]
#        bk_m = [31-e for e in range(m+1)]
#        fr_m.extend(bk_m)
#        data = data[:,fr_m]
    
        sum_acc = np.zeros((45,))
        kf = KFold(n_splits=45)
        lda = LinearDiscriminantAnalysis()
        i = 0
        for train_idx, test_idx in kf.split(data):
            lda.fit(data[train_idx], label[train_idx])
            score = int(lda.score(data[test_idx], label[test_idx]))
#            print("Test set score: %f" % score)
            sum_acc[i] = score
            i += 1
            
        average_acc = np.round(np.mean(sum_acc)*100, 3)
#        print("Total LOOCV accuracy: ", average_acc, '%')
        subs_acc[idx,m] = average_acc
#        print('===============================================')
