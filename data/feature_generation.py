#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:53:00 2019

@author: dominic
"""

import sklearn as sk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
#%matplotlib inline

######################
# READ IN DATA FILES #
######################
print("Reading in data files...")
blossom = pd.read_csv('blossoms.csv')
weather_plus = pd.read_csv('weather.csv')


##################################################
# CALCULATE DAY OF THE YEAR WHEN STAGES OCCURRED #
##################################################
print("Calculating day of the year when stages occurred...")
blossom['green_color_days'] = 0
blossom['florets_vis_days'] = 0
blossom['ext_florets_days'] = 0
blossom['pedun_elong_days'] = 0
blossom['puffy_white_days'] = 0
blossom['peak_bloom_days'] = 0

def days(year, month, day):
    dt = datetime.date(year, month, day)
    return (dt - datetime.date(year, 1, 1)).days + 1

for idx, row in blossom.iterrows():
    year = row[0]
    for c in range(1, 7):
        [month, day] = [int(i) for i in row[c].split('/')]
        blossom.iloc[idx, c+6] = days(year, month, day)


####################################
# GENERATE ALL FEATURES IN DATASET #
####################################

weather_plus['date'] = pd.to_datetime(weather_plus['date'])
weather_plus['day_count'] = weather_plus['date'].transform(lambda x: days(x.year, x.month, x.day))
weather_plus = weather_plus[weather_plus.day_count <= 150].copy()

# Only run on subset of data for dev purposes. Uncomment for full dataset
#weather_plus = weather_plus[((weather_plus['date'].dt.year == 2016) | (weather_plus['date'].dt.year == 2015)) & (weather_plus['day_count'] <= 120)].copy()

def calculate_if_occurred(row, col):
    if row.day_count == int(blossom[blossom.Year == row.date.year][col]):
        return 1
#     elif row.day_count > int(blossom[blossom.Year == row.date.year]['peak_bloom_days'])+1:
#         return 0
    else:
        return 0

binary_states = {
    'is_green_buds':'green_color_days',
    'is_florets_visible': 'florets_vis_days',
    'is_florets_extended': 'ext_florets_days',
    'is_peduncle_elongated': 'pedun_elong_days',
    'is_pufffy_white': 'puffy_white_days',
    'is_peak_bloom': 'peak_bloom_days'
}

for state in binary_states.keys():
    weather_plus[state] = weather_plus.apply(calculate_if_occurred, axis=1, args=(binary_states[state],))

for idx, row in weather_plus.iterrows():
    for state in binary_states:
        weather_plus.loc[idx, 'days_since_'+state] = max(int(0), int(row.day_count - blossom[blossom.Year == row.date.year][binary_states[state]]))

# Temperature conversion: C -> F
# C  |  F
# -------
# -5 | 23
#  0 | 32
#  5 | 41
# 10 | 50
# 15 | 59
# 20 | 68

temps = [23, 32, 41, 50, 59, 68]
time_windows = [1, 2, 4, 8, 16]

for idx, row in weather_plus.iterrows():
#    year = row.date.year
#    ytd = weather_plus[(weather_plus['date'].dt.year == year) & (weather_plus['date'] < row.date)]
    for temp in temps:
        weather_plus.loc[idx,'today_hrs_above_'+str(temp)+'F'] = 24 * max((row.tempHi - temp) - max(float(row.tempLo) - temp, 0), 0) / (row.tempHi - float(row.tempLo))
        weather_plus.loc[idx,'today_hrs_below_'+str(temp)+'F'] = 24 * max((temp - float(row.tempLo)) - max(temp - row.tempHi, 0), 0) / (row.tempHi - float(row.tempLo))
        for time_w in time_windows:
            weather_plus.loc[idx,'total_hrs_above_'+str(temp)+'F_'+str(time_w)+'_days'] = weather_plus['today_hrs_above_'+str(temp)+'F'].loc[idx - min(row.day_count, time_w) : idx - 1].sum()
            weather_plus.loc[idx,'total_hrs_below_'+str(temp)+'F_'+str(time_w)+'_days'] = weather_plus['today_hrs_below_'+str(temp)+'F'].loc[idx - min(row.day_count, time_w) : idx - 1].sum()
        weather_plus.loc[idx,'total_hrs_above_'+str(temp)+'F_year_to_date'] = weather_plus['today_hrs_above_'+str(temp)+'F'].loc[idx - row.day_count : idx - 1].sum()
        weather_plus.loc[idx,'total_hrs_below_'+str(temp)+'F_year_to_date'] = weather_plus['today_hrs_below_'+str(temp)+'F'].loc[idx - row.day_count : idx - 1].sum()

###################
# HISTORICAL RAIN #
###################
        
def convert_precip(row):
    if row.precip == "T":
        return 0
    else:
        return row.precip
weather_plus['precip_numerical'] = weather_plus.apply(convert_precip, axis=1).astype(float)

for idx, row in weather_plus.iterrows():
    year = row.date.year
    for time_w in time_windows:
        weather_plus.loc[idx,'total_precip_in_last_'+str(time_w)+'_days'] = weather_plus['precip_numerical'].loc[idx - min(row.day_count, time_w) : idx - 1].sum()


# TODO: make variables for future weather. In this case it will be
#        known perfectly, but real datasets will have to simulate it.

######################################
# CALCULATE FUTURE WEATHER VARIABLES #
######################################
time_windows = ((1,5), (5,10), (11,15), (16,20), (21, 25), (26, 30), (31, 35))
for idx, row in weather_plus.iterrows():
#    for future_day in range(10):
#        future_day = future_day + 1
#        weather_plus.loc[idx, 'future_tempHi_in_'+str(future_day)+'_days'] = weather_plus.loc[min(idx+future_day, idx-row.day_count+120), 'tempHi']
#        weather_plus.loc[idx, 'future_tempLo_in_'+str(future_day)+'_days'] = weather_plus.loc[min(idx+future_day, idx-row.day_count+120), 'tempLo']
#        weather_plus.loc[idx, 'future_tempAv_in_'+str(future_day)+'_days'] = weather_plus.loc[min(idx+future_day, idx-row.day_count+120), 'tempLo']
#        weather_plus.loc[idx, 'future_precip_in_'+str(future_day)+'_days'] = weather_plus.loc[min(idx+future_day, idx-row.day_count+120), 'precip_numerical']
    for temp in temps:
        for window in time_windows:
            above_name = 'future_hrs_above_'+str(temp)+'by'+str(window[1])+'_days'
            below_name = 'future_hrs_above_'+str(temp)+'by'+str(window[1])+'_days'
            timeslice = weather_plus.loc[idx+window[0]:idx+window[1]]
            hi = timeslice.tempHi
            lo = timeslice.tempLo.astype(float)
            above = sum(24 * ((hi - temp) - (lo - temp).apply(max, args=[0])).apply(max, args=[0]) / (hi-lo))
            below = sum(24 * ((temp - lo) - (temp - hi).apply(max, args=[0])).apply(max, args=[0]) / (hi-lo))
            weather_plus.loc[idx, above_name] = above
            weather_plus.loc[idx, below_name] = below


################################
# CALCULATE PREDICTOR VARIABLE #
################################
for idx, row in weather_plus.iterrows():
    year = row.date.year
    weather_plus.loc[idx, 'days_to_peak_bloom'] = int(blossom[blossom.Year == row.date.year]['peak_bloom_days']) - weather_plus.loc[idx, 'day_count']


#################################
# SAVE FEATURES AS TEST & TRAIN #
#################################
train = weather_plus[weather_plus['date'].dt.year <= 2013]
train.info()
train.to_csv('train.csv')
test = weather_plus[weather_plus['date'].dt.year > 2013]
test.info()
test.to_csv('test.csv')

#for idx, row in weather_plus.iterrows():
#    try:
#        float(row.tempLo)
#    except Exception as e:
#        print(idx, row.date, row.tempHi, row.tempLo)
##        break

# features to add:
#  - days since: florets, green buds
#  - hours above/below 0, 10, etc. for that day
#  - cumulative hours above/below 0, 10, etc. for year before that day
#  - rolling average of hours above/below 0, 10, etc. in past day, week, fortnight
#  - cumulative rain
#  - transform trace rain to 0 (or 0.01?)

# Y: days until peak bloom (negative after the event)

# metric:
#  - try penalizing differently, e.g. penalize closer days more heavily if they're wrong.


# How to predict?
#  final model is going to take in historical data + forecasted weather
#     - will have to simulate weather many times
#  

# How to display?
#  most likely date
#  timeline of events
#    -> below that, show historical variation of each event's prediction
