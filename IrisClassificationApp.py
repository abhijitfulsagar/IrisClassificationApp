# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:22:26 2018

@author: ABHIJIT
"""

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# loading the dataset
dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
X = dataset.iloc[:, 0:4]
Y = dataset.iloc[:,4]

#splitting the data set into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric= 'minkowski',p=2)
classifier.fit(X_train,Y_train)

yPred = classifier.predict(X_test)

print(accuracy_score(Y_test, yPred))
print(confusion_matrix(Y_test, yPred))
print(classification_report(Y_test, yPred))

