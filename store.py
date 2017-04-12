# coding=utf-8
__author__ = 'Pengfei'
from collections import Counter
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv

# load data
header = ['user_id', 'item_id', 'rating','timestamp']
df = pd.read_csv('data2.txt', sep='\t', names=header)
train_data, test_data = cv.train_test_split(df, test_size=0.2)

# Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((943,1682))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((943,1682))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

# Create W matrix
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = 1 - pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = 1 - pairwise_distances(train_data_matrix.T, metric='cosine')

Waa=np.zeros((943, 943))
Wbb=np.zeros((1682, 1682))
Wab=np.zeros((943, 1682))
for i in range(0,943):
    for j in range(0,943):
        Waa[i][j]=user_similarity[i][j]

for i in range(0,943):
    for j in range(0,1682):
        if train_data_matrix[i][j] != 0:
            Wab[i][j] = 1
# print Wab
Wba=Wab.T
W=np.bmat([[Waa, Wab], [Wba, Wbb]])
# print W
# print (train_data_matrix[1,1]-(np.sum(train_data_matrix[:,1])/(943-Counter(train_data_matrix[:,1])[0])))
# print (train_data_matrix[1,1]-np.mean(train_data_matrix[:,1]))