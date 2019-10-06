# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:28:24 2019

@author: aksmit94
"""

import pandas as pd
from datetime import datetime


def convert_unix_ts(df, timecols):
    """
    converts unix timestamp columns to human readable datetime
    :param df: imput dataframe
    :param timecols: list of columns containing unix timestamp
    :return: dataframe with unix timestamp columns converted to datetime
    """
    def convert(x):
        return datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S')

    for col in timecols:
        df[col] = df[col].apply(lambda x: convert(x))
        df[col] = pd.to_datetime(df[col])

    return df


def ride_duration(df):
    """
    Add column for duration of ride in minutes
    :param df: Input dataframe
    :return:
    """
    assert 'ride_start_timestamp' in list(df.columns) and 'ride_stop_timestamp' in list(df.columns)

    df['ride_duration'] = (df.ride_stop_timestamp - df.ride_start_timestamp).dt.total_seconds / 60

    return df


def driver_off_time():
    """
    Creates dataframe with driver_id, app-on duration and ride-service duration based on the logic
    that driver is not servicing a ride between timestamps

    :return:
    """
    return
