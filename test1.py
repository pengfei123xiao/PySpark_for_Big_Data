# coding=utf-8
__author__ = 'Pengfei'
import numpy as np
import network
import store
from rankclus import RankClus
from ranker import SimpleRanker
from sklearn.metrics import mean_squared_error
from math import sqrt

matrix =[store.Waa, store.Wab, store.Wba, store.Wbb]
ranker = SimpleRanker()

u_ID = [m for m in range(1,944)]
i_ID = [m for m in range(1,1683)]

net2 = network.Network(matrix, [u_ID,i_ID])
rc = RankClus(net2, ranker, 10)
rc.run()
rc.printResult(10)

all_pred = np.zeros((943, 1682))
f=open("allpred.txt",'w')
for i in range(0, 943):
    print i
    for j in range(0, 1682):
        all_pred[i][j] = rc.printpred(i, j, store.train_data_matrix, store.item_similarity)
        f.write(str(i))
        f.write(' ')
        f.write(str(j))
        f.write(' ')
        f.write(str(all_pred[i][j]))
        f.write('\n')
f.close()

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    fpr, tpr, thresholds = roc_curve(ground_truth, prediction, pos_label=2)
    return sqrt(mean_squared_error(prediction, ground_truth)),fpr,tpr

RMSE,x,y=rmse(all_pred, store.test_data_matrix)
print ("rmse is %f")%RMSE
f=open("rc-rmse1.txt",'w')
f.write(str(RMSE))
f.close()

print("AUC is %f")%auc(x,y)
import pylab as pl
pl.title("ROC curve of RC_ICF(AUC = %.4f)" % (auc(x,y)))
pl.xlabel("False Positive Rate")
pl.ylabel("True Positive Rate")
pl.plot(x, y,linestyle='dashed', color='orange', linewidth=2, label='ROC curve')  # use pylab to plot x and y
pl.show()  # show the plot on the screen
# personal idea:要想限制节点个数，rankclus,ranker,network内的limit都要改
