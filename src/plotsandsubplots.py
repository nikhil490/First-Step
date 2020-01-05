# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris() # returns a bunch object which is similar to the dictionary with keys and values for eg the descr key point to the descriptioon of the data, data point tió the 150 X 1 data matrix etc acess the data by bunch.key
x_train,x_test,y_train,y_test = train_test_split(iris['data'],iris['target'],random_state = 0)
#train_test_split : it splits the data into train and test. random_state= 0 implies that the splitting is happening in a similar way whenever we run the code
fig,ax = plt.subplots(3,3,figsize = (15,15))
plt.suptitle("iris_pairplot")
for i in range(3):
    for j in range(3):
        ax[i,j].scatter(x_train[:,j],x_train[:,i+1],c = y_train, s=60)
        #ax[i,j].set_xticks(())
        #ax[i,j].set_yticks(())
        if i == 2:
            ax[i, j].set_xlabel(iris['feature_names'][j])
        if j == 0:
            ax[i, j].set_ylabel(iris['feature_names'][i + 1])
        if j > i:
            ax[i, j].set_visible(False) 

knn = KNeighborsClassifier(n_neighbors = 1) # the classifier take the test data model the data based on the ALgorithm
knn.fit(x_train,y_train) # training the data
KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')