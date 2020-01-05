# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 18:42:12 2020

@author: Nikhil
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris = load_iris()
x_train,x_test,y_train,y_test = train_test_split(iris['data'],iris['target'],random_state = 0)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)
KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')
x_new = np.array([[5,2.9,1,0.2]])
prediction = knn.predict(x_new)
knn.score(x_test,y_test)
