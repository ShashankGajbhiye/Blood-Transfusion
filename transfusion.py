# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:20:52 2019

@author: Shashank
"""

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the Dataset
data = pd.read_csv('transfusion.data', sep = ',')
dataset = pd.read_csv('transfusion.data', sep = ',')
dataset.isnull().sum()
dataset.dtypes
dataset = dataset.rename(columns = {'Recency (months)' : 'Recency',
                                    'Frequency (times)' : 'Frequency',
                                    'Monetary (c.c. blood)' : 'Monetary',
                                    'Time (months)' : 'Time',
                                    'whether he/she donated blood in March 2007': 'Donated'})
x = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# Exploring the Dataset
dataset['Recency'].value_counts()
dataset[(dataset['Donated'] == 1)]['Recency'].plot(kind = 'bar')
sns.distplot(dataset['Donated'])
dataset['Donated'].value_counts()

# Splitting the Dataset into Training And Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.331, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Fitting the XGBoost Classification to The Training Set
from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimators = 50, gamma = 0.3)
classifier.fit(x_train, y_train)

"""
# Fitting the Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
"""

# Predicting the Test Set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying the K-fold Cross Validation 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying the Grid Search Model to Classifier
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [10, 30, 50, 100], 'gamma' : [0.1, 0.05, 0.3, 0.5, 1]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters,
                           scoring = 'accuracy', n_jobs = -1, cv = 10)
grid_search = grid_search.fit(x_train, y_train)

best_acc = grid_search.best_score_
best_parameter = grid_search.best_params_
print('Best Accuracy:', best_acc)
print('Best Parameters:', best_parameter)
