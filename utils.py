# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:28:24 2019

@author: aksmit94
"""

import pandas as pd
<<<<<<< HEAD
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

    df['ride_duration'] = (df.ride_stop_timestamp - df.ride_start_timestamp).dt.total_seconds() / 60

    return df


def driver_off_time():
    """
    Creates dataframe with driver_id, app-on duration and ride-service duration based on the logic
    that driver is not servicing a ride between timestamps

    :return:
    """
    return


def diff_between_rides(df):
    """
    Calculating the active time
    :param df:
    :return:
    """
    df_new = df.copy()
    df_new.sort_values(['driver_id', 'ride_start_timestamp'], inplace=True)
    df_new['stop_time_shifted'] = df_new.groupby('driver_id')['ride_stop_timestamp'].shift(1)
    df_new['diff'] = df_new['ride_start_timestamp'] - df_new['stop_time_shifted']

    return df_new

def idle_time_calculation(df, threshold = 3600, colname='diff'):
    """

    :param df:
    :param threshold:
    :return:
    """
    df_new = df[df[colname].dt.total_seconds() > 2 * threshold]
    total_inactive_time = df_new.groupby('driver_id')[colname].sum().reset_index()
    total_inactive_time['inactive_time'] = total_inactive_time['diff'].dt.total_seconds() / 60
=======
import utm
import os
import numpy as np

from common import CACHE_DIR


def get_start_end_bins(df, cols):
    #    date_curr = pd.to_datetime(date)
    #    year = pd.to_datetime(date).year
    #    month = pd.to_datetime(date).month
    #    day = pd.to_datetime(date).day

    for col in cols:
        df[f'{col}_only_time'] = df[f'{col}'] - df[f'{col}'].dt.normalize()
        #        breakpoint()
        df[f'{col}_bin'] = pd.qcut(df[f'{col}_only_time'], 4)


def get_spatial_features(df, grid_x_num=40, grid_y_num=40):
    #    cache_path = os.path.join(CACHE_DIR, f'merged_orders.msgpack')
    #    if os.path.exists(cache_path):
    #        print(f'{cache_path} exists')
    #        df_new = pd.read_msgpack(cache_path)
    #    else:

    temp_pickup = df.apply(
        lambda x: utm.from_latlon(x['pickup_latitude'], x['pickup_longitude'])[
            0:2],
        axis=1)
    temp_drop = df.apply(
        lambda x: utm.from_latlon(x['dropoff_latitude'], x['dropoff_longitude']
                                  )[0:2],
        axis=1)

    df['xpickup'] = temp_pickup.str[0]
    df['ypickup'] = temp_pickup.str[1]

    df['xdropoff'] = temp_drop.str[0]
    df['ydropoff'] = temp_drop.str[1]

    #    breakpoint()

    tempx = pd.cut(df['xpickup'], bins=grid_x_num).astype(str)
    tempy = pd.cut(df['ypickup'], bins=grid_y_num).astype(str)
    df['pick_up_zone'] = tempx + tempy

    tempx = pd.cut(df['xdropoff'], bins=grid_x_num).astype(str)
    tempy = pd.cut(df['ydropoff'], bins=grid_y_num).astype(str)
    df['drop_off_zone'] = tempx + tempy

    #    temp = (df.groupby(['driver_id', 'pick_up_zone']).count() / df.groupby(
    #        ['driver_id']).count()).unstack(level=1)

    #    breakpoint()
    grouped_tmp = df[['driver_id', 'pick_up_zone', 'pickup_latitude']].groupby(
        ['driver_id', 'pick_up_zone']).count() / df[[
            'driver_id', 'pick_up_zone', 'pickup_latitude'
        ]].groupby(['driver_id'])[['pickup_latitude']].count()
    #    breakpoint()
    #    grouped_tmp = grouped_tmp[['dropoff_latitude']]

    #    df1 = grouped_tmp.unstack(level=1)
    temp = grouped_tmp.unstack(level=0).T
    #    df1m = grouped_tmp.unstack(level=-1)

    #    (df.groupby(['driver_id', 'pick_up_zone']).count() / df.groupby(
    #        ['driver_id']).count())

    temp.fillna(0, inplace=True)
    return temp


def create_modified_active_time(orders):
    driver_start_times = orders.loc[:, [
        'driver_id', 'ride_start_timestamp', 'ride_stop_timestamp', 'order_id'
    ]].drop_duplicates()
    driver_start_times.sort_values(['driver_id', 'ride_start_timestamp'],
                                   inplace=True)
    driver_start_times['stop_time_shifted'] = driver_start_times.groupby(
        'driver_id')['ride_stop_timestamp'].shift(1)
    driver_start_times['diff'] = driver_start_times[
        'ride_start_timestamp'] - driver_start_times['stop_time_shifted']

    ##
    driver_day_min = pd.DataFrame(
        orders.groupby('driver_id')
        ['ride_start_timestamp'].min()).reset_index()
    driver_day_max = pd.DataFrame(
        orders.groupby('driver_id')
        ['ride_stop_timestamp'].max()).reset_index()
    driver_active_time = driver_day_min.merge(
        driver_day_max, on='driver_id', how='left')
    driver_active_time['active_time'] = (
        driver_active_time['ride_stop_timestamp'] -
        driver_active_time['ride_start_timestamp']).dt.total_seconds() / 60
    driver_active_time.head()
    ##

    ##
    driver_ride_durations = orders.groupby('driver_id')[[
        'ride_duration'
    ]].sum().reset_index()
    ##

    ##
    # total driver active time
    driver_stats = driver_active_time[['driver_id', 'active_time']].merge(
        driver_ride_durations, on='driver_id', how='left')

    # Filtering possible bad rows
    driver_stats[
        driver_stats.active_time < driver_stats.ride_duration].describe()
    ##

    ##
    drivers_with_greater_than_an_hour_break = driver_start_times[
        driver_start_times['diff'].dt.total_seconds() > 3600]
    total_inactive_time = drivers_with_greater_than_an_hour_break.groupby(
        'driver_id')['diff'].sum().reset_index()
    total_inactive_time[
        'inactive_time'] = total_inactive_time['diff'].dt.total_seconds() / 60
    ##

    ##
    driver_stats_updated = driver_stats.merge(
        total_inactive_time[['driver_id', 'inactive_time']],
        on='driver_id',
        how='left').fillna(0)
    driver_stats_updated['modified_active_time'] = driver_stats_updated[
        'active_time'] - driver_stats_updated['inactive_time']
    driver_stats_updated[
        'modified_active_time_with_rules'] = driver_stats_updated[
            'active_time'] - np.where(
                (driver_stats_updated['inactive_time'] > 240) |
                (driver_stats_updated['inactive_time'] == 0),
                driver_stats_updated['inactive_time'], 60)
    ##

    cols = [
        'driver_id', 'ride_duration', 'modified_active_time',
        'modified_active_time_with_rules'
    ]

    return driver_stats_updated[cols]
>>>>>>> 9e312e1027e965a313dbbf343530904e7eb4ff80
