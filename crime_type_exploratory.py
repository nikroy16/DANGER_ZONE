import numpy as np
import pandas as pd
from sklearn import preprocessing
from datetime import datetime
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


#List of all features available to use: 'POINT_X', 'POINT_Y', 'DATE', 'WEEK_DAY', 'MONTH', HOUR', 'TEMP, 'WIND', 'WIND_DIR', 'S_RAIN', 'S_SNOW', 'HUMIDITY', 'VISIBILITY', 'CLOUD COVER', 'PRECIP_INCHES', 'TEMP_LABELED', 'WIND_BINARY'  'YEAR'


data = pd.read_pickle('data/data_processed.pkl')
features = ['TEMP_LABELED', 'TEMP_LABELED','HOUR']
specific_data = data[features + ['CRIME_TYPE']]

X = np.array(specific_data)



### only use the code below if algorithm requires normalization before hand ###
'''
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
'''



y = X[:,len(features)]

X= np.delete(X, len(features), 1)








predictions = []
actual = []
acc = []

kf = cross_validation.KFold(len(y), n_folds=10)

    
for train_index, test_index in kf:
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        
        
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        
        pred = clf.predict(X_test)
        accuracyDT = accuracy_score(y_test, pred)
        
        acc.append(accuracyDT)






print 'Accuracy', np.mean(acc)

print(metrics.classification_report(y_test, pred))