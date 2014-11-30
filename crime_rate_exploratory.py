import numpy as np
import pandas as pd
from sklearn import preprocessing
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
import os
import subprocess

mtr = 'data/month_temp_rate.pkl'
hwdr = 'data/hour_wind_day_rate.pkl'

filepath = mtr

#Read in data
data = pd.read_pickle(filepath)

#Binarize rate
median = np.median(data['RATE'])
def make_binary(x):
    if x >= median:
        return 1
    else:
        return 0

b = np.vectorize(make_binary)

data['RATE'] = b(data['RATE'])

# Build numpy array
X = np.array(data)

### only use the code below if algorithm requires normalization before hand ###
'''
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
'''
n,d = X.shape

# Retrieve class label (rate)
y = X[:,d - 1]

# Delete class label and year feature
X= np.delete(X, d - 1, 1)
X= np.delete(X, 0,1)


pred = np.array([])
actual = np.array([])




kf = cross_validation.KFold(len(y), n_folds=10)

counter = 0   
for train_index, test_index in kf:
        print counter
        counter +=1
        
        
        
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        
        
        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        pred = np.append(pred,clf.predict(X_test))
        
        actual = np.append(actual, y_test)

print 'Accuracy', metrics.accuracy_score(actual, pred)
print(metrics.classification_report(actual, pred))
