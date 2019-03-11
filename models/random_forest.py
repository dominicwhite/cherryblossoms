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
import pickle


data = pd.read_csv("../train.csv", index_col=0)
data1 = data[(data.day_count > 30) & (data.days_to_peak_bloom >= 0)]

parameter_columns = [col for col in data1.columns]
non_parameter_columns = ['date', 'precip','days_to_peak_bloom']
for npc in non_parameter_columns:
    parameter_columns.remove(npc)

X = data1[parameter_columns]
y = data1['days_to_peak_bloom']

rfr = RandomForestRegressor()
search = RandomizedSearchCV(
        
        rfr, 
        param_distributions={
                'n_estimators': [4,8,16,32,64,128,256],
                'max_features': ['auto', 'log2', 'sqrt']
        },
        n_iter = 100,
#        cv = TimeSeriesSplit
        )
search.fit(X, y)

scores = [ (s, p) for s, p in zip(parameter_columns, search.best_estimator_.feature_importances_)]
scores = sorted(scores, key = lambda x: x[1], reverse=True)
for p, s in scores:
    print(p, s)

train_preds = search.best_estimator_.predict(X)

with open('rf_out.sav', 'wb') as f:
    pickle.dump(search.best_estimator_, f)
