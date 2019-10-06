# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:28:24 2019

@author: aksmit94
"""

import pandas as pd
import datetime

def get_start_end_bins(df, date, cols):
    date_curr = pd.to_datetime(date)
    year = pd.to_datetime(date).year
    month = pd.to_datetime(date).month
    day = pd.to_datetime(date).day
    
    #TODO remove hard coding
    bins = [
        (datetime.datetime(year, month, day, 00, 0, 0) - date_curr),
        (datetime.datetime(year, month, day, 10, 30, 0) - date_curr),
        (datetime.datetime(year, month, day, 14, 30, 0) - date_curr),
        (datetime.datetime(year, month, day, 18, 30, 0) - date_curr),
        (datetime.datetime(year, month, day, 23, 59, 59) - date_curr)
    ]
    for col in cols:
        df[f'{col}_only'] = df[f'{col}'] - date_curr
        df[f'{col}_bin'] = pd.cut(x=df[f'{col}_only'], bins=bins)

def driver_off_time():
    """
    Creates dataframe with driver_id, app-on duration and ride-service duration based on the logic
    that driver is not servicing a ride between timestamps

    :return:
    """
    return
