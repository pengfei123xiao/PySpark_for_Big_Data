# coding=utf-8
import numpy as np
import store
from sklearn.metrics import mean_squared_error
from math import sqrt


u_ID = [m for m in range(1,944)]
i_ID = [m for m in range(1,1683)]

# net2 = network.Network(matrix, [u_ID,i_ID])
# rc = RankClus(net2, ranker, 10)
# rc.run()
# rc.printResult(10)

def i_predict(u_ID, i_ID, ratings, similarity):
    up = 0.0
    down = 0.0
    for i in range(943):
        up += similarity[u_ID - 1, i] * ratings[i,i_ID]
        # up += 0.5 * similarity[i_ID - 1, i] * ratings[u_ID-1, i]
        # print self.conrankY[n][i] * similarity[i_ID - 1, i] * ratings_diff
        # down = abs(score[u_ID - 1] * similarity[u_ID - 1, i])
        down += similarity[u_ID - 1, i]
    if down == 0:
        res = 0
    else:
        res = up / down
    pred =res
    # print 1
    return pred

all_pred = np.zeros((943, 1682))
for i in range(0, 943):
    for j in range(0, 1682):
        all_pred[i][j] = i_predict(i, j, store.train_data_matrix, store.user_similarity)
        print i

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    fpr, tpr, thresholds = roc_curve(ground_truth, prediction, pos_label=2)
    return sqrt(mean_squared_error(prediction, ground_truth)),fpr,tpr

RMSE,x,y=rmse(all_pred, store.test_data_matrix)
print ("rmse is %f")%RMSE
f=open("rmse3-ucf.txt",'w')
f.write(str(RMSE))
f.close()

print("AUC is %f")%auc(x,y)
import pylab as pl
pl.title("ROC curve for UCF (AUC = %.4f)" % auc(x,y))
pl.xlabel("False Positive Rate")
pl.ylabel("True Positive Rate")
pl.plot(x, y,linestyle='dashed', color='orange', linewidth=2, label='ROC curve')  # use pylab to plot x and y
pl.show()  # show the plot on the screen
# personal idea:要想限制节点个数，rankclus,ranker,network内的limit都要改