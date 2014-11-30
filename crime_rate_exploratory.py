import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB

#Available datasets: month_temp_rate.pkl, hour_wind_day_rate.pkl

mtr = 'data/month_temp_rate.pkl'
hwdr = 'data/hour_wind_day_rate.pkl'

filepath = mtr

#Retrieve data, split into instance/label

data = pd.read_pickle(filepath)
X = np.array(data)
print X.shape
n,d = X.shape
y = X[:, d - 1]
X = np.delete(X, d - 1, 1)
print X.shape
print y.shape

predictions = []
actual = []
acc = []

kf = cross_validation.KFold(len(y), n_folds=2)

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
