#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 07:10:32 2019

@author: dominic
"""
import csv
import datetime
import os
import pandas as pd
import requests

KEY = os.environ.get("DARKSKY_KEY")
tidal_basin_lat = 38.883995
tidal_basin_long = -77.038976


def format_darksky_url(dtime, lat=38.883995, long=-77.038976):
    time = dtime.strftime("%Y-%m-%dT%H:%M:%S")
    print(time)
    api_base = f"https://api.darksky.net/forecast/{KEY}/{lat},{long},{time}"
    get_params = "?exclude=hourly"
    print(api_base + get_params)
    return api_base + get_params


def parse_darksky_response(response_json):
    data = {}
    data["tempHi"] = response_json["temperatureHigh"]
    data["tempLo"] = response_json["temperatureLow"]
    data["tempAv"] = data["tempLo"] + (data["tempHi"] - data["tempLo"]) / 2
    return data

#print(parse_darksky_response(rj))

def save_2019_data():
    one_day = datetime.timedelta(days=1)
    start2019 = datetime.datetime(2019, 1, 1, 12)
#    print(start2019.strftime("%Y-%m-%dT%H:%M:%S"))
    rdate = start2019
    hist_data2019 = []
    while rdate < datetime.datetime.now():
        endpoint = format_darksky_url(rdate)
        print(rdate)
        r = requests.get(endpoint)
        rj = r.json()
        temps = parse_darksky_response(rj["daily"]["data"][0])
        temps['date'] = rdate.strftime("%Y-%m-%d")
        hist_data2019.append(temps)
        rdate += one_day
#    print(hist_data2019)
    with open("data/2019_weather_hist.csv", "w") as f:
        writer = csv.DictWriter(f, ["date", "tempHi", "tempLo", "tempAv"])
        writer.writeheader()
        writer.writerows(hist_data2019)


#save_2019_data()


def save_forecast():
    lat = 38.883995
    long = -77.038976
    api_base = f"https://api.darksky.net/forecast/{KEY}/{lat},{long}"
    get_params = "?exclude=hourly"
    endpoint = api_base + get_params
    print(endpoint)
    r = requests.get(endpoint)
    rj = r.json()
    forecast = []
    for day in rj["daily"]["data"]:
        temps = parse_darksky_response(day)
        day_dt = datetime.datetime.fromtimestamp(day["time"])
        temps['date'] = day_dt.strftime("%Y-%m-%d")
        forecast.append(temps)
    todays_date = datetime.datetime.now().strftime("%Y-%m-%d")
    with open("data/forecast_" + todays_date + ".csv", "w") as f:
        writer = csv.DictWriter(f, ["date", "tempHi", "tempLo", "tempAv"])
        writer.writeheader()
        writer.writerows(forecast)

save_forecast()
