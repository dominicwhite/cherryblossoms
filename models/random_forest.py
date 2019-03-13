#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:17:50 2019

@author: dominic
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pickle


train_data = pd.read_csv("../data/train.csv", index_col=0)
test_data = pd.read_csv("../data/test.csv", index_col=0)
parameter_columns = [col for col in train_data.columns]

def transform_data(data, cols, days_past_peak = 0):
    data1 = data[(data.day_count > 30) & (data.days_to_peak_bloom >= -days_past_peak)]
    
    parameter_columns = [col for col in data1.columns]
    non_parameter_columns = ['date', 'precip','days_to_peak_bloom']
    for npc in non_parameter_columns:
        parameter_columns.remove(npc)
    
    X = data1[parameter_columns]
    y = data1['days_to_peak_bloom']
    return X, y, parameter_columns


####################
# 0 days past peak #
####################
X_train, y_train, parameter_columns = transform_data(train_data, parameter_columns)
rfr = RandomForestRegressor()
search = RandomizedSearchCV(
        rfr, 
        param_distributions={
                'n_estimators': [4,8,16,32,64,128,256],
                'max_features': ['auto', 'log2', 'sqrt']
        },
        n_iter = 100
        )
search.fit(X_train, y_train)

scores = [ (s, p) for s, p in zip(parameter_columns, search.best_estimator_.feature_importances_)]
scores = sorted(scores, key = lambda x: x[1], reverse=True)
for p, s in scores:
    print(p, s)

train_preds = search.best_estimator_.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
print(train_rmse)

with open('rf_out.sav', 'wb') as f:
    pickle.dump(search.best_estimator_, f)


X_test, y_test, _ = transform_data(test_data, parameter_columns)
test_preds = search.best_estimator_.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
print(test_rmse)

plt.figure(1)
plt.scatter(y_test, test_preds)
plt.show()

plt.figure(2)
plt.scatter(y_test, np.abs(y_test-test_preds))
plt.show()

# Has train and test errors of 0.678 and 5.517



#####################
# 20 days past peak #
#####################
X_train20, y_train20, _ = transform_data(train_data, parameter_columns, days_past_peak = 20)
#rfr = RandomForestRegressor()
search20 = RandomizedSearchCV(
        rfr, 
        param_distributions={
                'n_estimators': [4,8,16,32,64,128,256],
                'max_features': ['auto', 'log2', 'sqrt']
        },
        n_iter = 100
        )
search20.fit(X_train20, y_train20)

#scores = [ (s, p) for s, p in zip(parameter_columns, search20.best_estimator_.feature_importances_)]
#scores = sorted(scores, key = lambda x: x[1], reverse=True)
#for p, s in scores:
#    print(p, s)

train_preds20 = search20.best_estimator_.predict(X_train20)
train_rmse20 = np.sqrt(mean_squared_error(y_train20, train_preds20))
print(train_rmse20)

with open('rf_out20.sav', 'wb') as f:
    pickle.dump(search20.best_estimator_, f)


X_test20, y_test20, _ = transform_data(test_data, parameter_columns, days_past_peak = 20)
test_preds20 = search20.best_estimator_.predict(X_test20)
test_rmse20 = np.sqrt(mean_squared_error(y_test20, test_preds20))
print(test_rmse20)

plt.figure(1)
plt.scatter(y_test20, test_preds20)
plt.show()

plt.figure(2)
plt.scatter(y_test20, np.abs(y_test20-test_preds20))
plt.show()

## Has train and test RMSE of 0.613 and 4.760

