import math
from collections import OrderedDict
from operator import itemgetter
from numpy import genfromtxt
import numpy as np

def d(Xtest,Xtrain):
    sum=0
    return np.linalg.norm(Xtest-Xtrain)

k=10
train=genfromtxt("MNIST_train_small.csv",delimiter=',')
test=genfromtxt("MNIST_test_small.csv",delimiter=',')
distancelist=[]
# print(test[0][1:])
mis_match=0
for i in range(len(test)):
    distancelist=[]
    for j in range(len(train)): # for every test data, calculate distance to all training data
        distancelist.append([train[j][0],d(test[i][1:],train[j][1:])])
    distancelist=sorted(distancelist, key=itemgetter(1))
    # print(distancelist)
    first_k=[]
    for j in range(k): # take first kth smallest distances
        first_k.append(distancelist[j][0])
    prediction=max(first_k, key = first_k.count)
    if prediction != test[i][0]:
        mis_match+=1
        print(prediction, test[i][0])
print(f'accuracy:{1-mis_match/len(test)}')

