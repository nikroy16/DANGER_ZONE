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

data = pd.read_pickle('data/month_temp_rate.pkl')
median = np.median(data['RATE'])
def make_binary(x):
    if x >= median:
        return 1
    else:
        return 0

b = np.vectorize(make_binary)

data['RATE'] = b(data['RATE'])




features = ['MONTH', 'TEMP_LABELED']

specific_data = data[features + ['RATE']]

X = np.array(specific_data)



### only use the code below if algorithm requires normalization before hand ###
'''
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
'''

y = X[:,len(features)]

X= np.delete(X, len(features), 1)


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
