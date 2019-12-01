# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:28:24 2019

@author: aksmit94
"""
#!pip install folium
#!pip install utm

import math
import pandas as pd
import utm
import os
import numpy as np
import matplotlib.pyplot as plt

from h3 import h3
from common import (CACHE_DIR, X_CENTER, Y_CENTER,
                    NUM_CUTS_R_RADIAL, NUM_CUTS_THETA_RADIAL)
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def get_start_end_bins(df, cols):
    for col in cols:
        df[f'{col}_only_time'] = df[f'{col}'] - df[f'{col}'].dt.normalize()
        df[f'{col}_bin'] = pd.qcut(df[f'{col}_only_time'], 4)


def idle_time_est(t, tau, shape):
    # sample from time diff distribution
    t_s = tau + np.random.exponential(scale=shape)

    indicator = t <= t_s
    indicator = 1 - indicator
    i_time = t - t_s
    idle_time = indicator * i_time

    return idle_time


def idle_time_est_old(t, tau, shape, size):
    # sample from time diff distribution
    t_s = tau + np.random.exponential(scale=shape, size=size)

    indicator = t <= t_s
    indicator = 1 - indicator
    i_time = t - t_s
    idle_time = indicator * i_time

    return idle_time


def get_inv_cdf(x, c, x0=10):
    norm_value = c/np.log(1+np.exp(c*x0))
    updated_c = c/norm_value
    # val = np.exp(c*(1-x))
    val = np.exp(updated_c*(1-x))
    inv_y = (np.log(val / (1 + np.exp(c*x0) - val)) / c) + x0
    return inv_y


def pool_rides(orders):
    """
    Create column for number of pool rides for a driver
    """

    orders.sort_values(by=['driver_id', 'ride_start_timestamp'], inplace=True)
    orders['shifted_end_time'] = orders['ride_stop_timestamp'].shift(1)
    orders['shifted_driver'] = orders['driver_id'].shift(1)
    orders[
        'cond_1'] = orders['ride_start_timestamp'] < orders['shifted_end_time']
    orders['cond_2'] = orders['shifted_driver'] == orders['driver_id']
    orders['is_pool'] = (orders['cond_1'] & orders['cond_2'])


def group_pool(df):
    df = df.copy()
    pool_rides(df)
    df['is_pool_shifted'] = df.groupby('driver_id')['is_pool'].shift(-1).fillna(False)
    df['pool'] = df['is_pool'] + df['is_pool_shifted']
    # df['pool_shifted_down'] =  df.groupby('driver_id')['pool'].shift(1)
    # df['first_pool_1'] = np.where(df['pool'] != df['pool_shifted_down'], 1, 0)
    # df['first_pool_1'].fillna(1)
    df['factor'] = -df.index
    df['factor'][df.pool] = 1
    df['shifted_stop_time_for_ride'] = df.groupby('driver_id')['ride_stop_timestamp'].shift(1)
    df['check_shift_sign_updated'] = df['ride_start_timestamp'] - df['shifted_stop_time_for_ride']
    df['check_shift_sign'] = df['ride_start_timestamp'] < df['shifted_stop_time_for_ride']
    df['check_shift_sign_new'] = np.where(df['check_shift_sign_updated'].dt.total_seconds() < 0, 0, 1)
    df['check_shift_sign_new'].fillna(1)
    df['cum_sum_1'] = df['check_shift_sign'].cumsum()
    # df['cum_sum_1'] = df.groupby('driver_id')['check_shift_sign_new'].cumsum()
    df['group_1'] = np.where(df['pool'], df['cum_sum_1'], df['factor'])
    df['group_1_shifted_down'] = df['group_1'].shift(1)
    df['equal_column'] = df['group_1'] !=  df['group_1_shifted_down']
    df['cum_sum_2'] = df['equal_column'].cumsum()
    # df.iloc[~df.pool, 'factor'] = -df.iloc[~df.pool, 'index']
    # df.iloc[df.pool, 'factor'] = df.iloc[df.pool, int(df.is_pool)]
    # df['actual_ride_start'] = df.groupby('cum_sum_2')['ride_start_timestamp'].min().transform('min')
    df = df.groupby(['cum_sum_2']).agg({'ride_start_timestamp': min,
                                        'ride_stop_timestamp': max,
                                        'driver_id': 'first', 'order_id': 'first'}).reset_index()
    df['ride_duration'] = (df.ride_stop_timestamp -
                           df.ride_start_timestamp).dt.total_seconds() / 60
    return df


def get_surv_prob(orders, c=1, use_cache=True, combine_pool=False, save_file=True):
    cache_path = os.path.join(CACHE_DIR, f'survival_probability_df_pool_{combine_pool}.msgpack')
    if os.path.exists(cache_path) and use_cache:
        print(f'{cache_path} exists')
        driver_stats_updated = pd.read_msgpack(cache_path)
        return driver_stats_updated

    else:

        print("herex`")

        if combine_pool:
            orders = orders.copy()
            orders = group_pool(orders)

        print("Creating the Survival Functions")
        driver_start_times = orders.loc[:, ['driver_id', 'ride_start_timestamp',
                                            'ride_stop_timestamp', 'order_id']] \
            .drop_duplicates()
        driver_start_times.sort_values(['driver_id', 'ride_start_timestamp'],
                                       inplace=True)
        driver_start_times['stop_time_shifted'] = driver_start_times.groupby(
            'driver_id')['ride_stop_timestamp'].shift(1)
        driver_start_times['diff'] = driver_start_times[
                                         'ride_start_timestamp'] - \
                                     driver_start_times['stop_time_shifted']

        driver_start_times_no_na = driver_start_times.dropna()
        driver_start_times_no_na['diff'] = driver_start_times_no_na[
                                               'diff'].dt.total_seconds() / 60

        rand_val = np.random.random(size=len(driver_start_times_no_na))
        driver_start_times_no_na['survival_active_time'] = get_inv_cdf(rand_val, c)
        driver_start_times_no_na['survival_active_time'] = np.minimum(np.maximum(driver_start_times_no_na['diff'], 0),
                                                                      driver_start_times_no_na['survival_active_time'])

        breakpoint()

        """
        When ride difference is negative, we are getting values where survival time is negative which reduces the sum, 
        So if the ride difference is negative, that would imply overlapping rides which would mean mean the driver 
        is active on the system for this ride, so we would have counted that ride for the pool and will take this 
        as 0  
        """

        """
        Our survival active time only considers the time the driver was active in between rides, it does not take 
        into account the those times into account when the driver was using in a ride. So our actual active time 
        will be the sum of the ride duration and the survival active time 
        
        So note this correction 
        """

        driver_day_min = pd.DataFrame(
            orders.groupby('driver_id')
            ['ride_start_timestamp'].min()).reset_index()
        driver_day_max = pd.DataFrame(
            orders.groupby('driver_id')
            ['ride_stop_timestamp'].max()).reset_index()
        driver_active_time = driver_day_min.merge(
            driver_day_max, on='driver_id', how='left')
        driver_active_time['active_time'] = (
                                                    driver_active_time[
                                                        'ride_stop_timestamp'] -
                                                    driver_active_time[
                                                        'ride_start_timestamp']).dt.total_seconds() / 60
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

        ##
        total_active_time = driver_start_times_no_na.groupby(
            'driver_id')['survival_active_time'].sum().reset_index()
        ##
        driver_stats_updated = driver_stats.merge(
            total_active_time[['driver_id', 'survival_active_time']],
            on='driver_id',
            how='left')

        # print(driver_stats_updated[driver_stats_updated.driver_id == '00002724a19c5f6a54ae8d60a378997e'])
        # exit(0)

        driver_stats_updated['survival_active_time'] = driver_stats_updated['survival_active_time'] + driver_stats_updated['ride_duration']

        driver_stats_updated.loc[(pd.isnull(driver_stats_updated.survival_active_time)),
                                 'survival_active_time'] = driver_stats_updated.active_time


        driver_stats_updated['inactive_time'] = driver_stats_updated['active_time'] - driver_stats_updated['survival_active_time']

        # print(driver_stats_updated[driver_stats_updated.driver_id == '00002724a19c5f6a54ae8d60a378997e'])
        # breakpoint()

        cols = [
            'driver_id', 'ride_duration', 'survival_active_time', 'inactive_time'
        ]

        driver_stats_updated = driver_stats_updated[cols]
        if save_file:
            pd.to_msgpack(cache_path, driver_stats_updated)
        print(f'Dumping to {cache_path}')


        return driver_stats_updated


# def get_tau_nought(c):
#     def f(c, x):
#         1/(1+math.exp(c*x))
#     tau0 = np.ln(1/0.95 - 1)/c
#     return tau0


def get_inverse_func(c, y):
    tau0 = np.ln(1 / 0.95 - 1) / c
    temp = np.ln(1/y - 1)/c
    return np.minimum(temp, tau0)


def get_spatial_features(df, grid_x_num=10, grid_y_num=10,
                         use_cache=True):
    cache_path = os.path.join(CACHE_DIR, f'spatial_df.msgpack')
    if os.path.exists(cache_path) and use_cache:
        print(f'{cache_path} exists')
        temp = pd.read_msgpack(cache_path)
    else:
        pickup_coord = utm.from_latlon(df['pickup_latitude'].values, df['pickup_longitude'].values)
        col1, col2 = pickup_coord[0], pickup_coord[1]
        df['xpickup'] = col1
        df['ypickup'] = col2

        dropoff_coord = utm.from_latlon(df['dropoff_latitude'].values, df['dropoff_longitude'].values)
        col3, col4 = dropoff_coord[0], dropoff_coord[1]
        df['xdropoff'] = col3
        df['ydropoff'] = col4

        tempx = pd.cut(df['xpickup'], bins=grid_x_num).astype(str)
        tempy = pd.cut(df['ypickup'], bins=grid_y_num).astype(str)
        df['pick_up_zone'] = tempx + tempy

        tempx = pd.cut(df['xdropoff'], bins=grid_x_num).astype(str)
        tempy = pd.cut(df['ydropoff'], bins=grid_y_num).astype(str)
        df['drop_off_zone'] = tempx + tempy

        grouped_tmp = df[['driver_id', 'pick_up_zone', 'pickup_latitude']].groupby(
            ['driver_id', 'pick_up_zone']).count() / df[[
                'driver_id', 'pick_up_zone', 'pickup_latitude'
            ]].groupby(['driver_id'])[['pickup_latitude']].count()
        temp = grouped_tmp.unstack(level=0).T
        temp.fillna(0, inplace=True)
        temp.reset_index(inplace=True)
        temp.drop(columns=['level_0'], inplace=True)
        pd.to_msgpack(cache_path, temp)
        print(f'Dumping to {cache_path}')
    return temp


def get_spatial_features_radial(df, grid_x_num=10, grid_y_num=10,
                         use_cache=True):
    cache_path = os.path.join(CACHE_DIR, f'radial_spatial_df.msgpack')
    if os.path.exists(cache_path) and use_cache:
        print(f'{cache_path} exists')
        temp = pd.read_msgpack(cache_path)
    else:
        cols = ['r_radial', 'theta_radial']
        create_radial_bins(df, cols)

        grouped_tmp = df[['driver_id', 'pick_up_radial_zone', 'pickup_latitude']].groupby(
            ['driver_id', 'pick_up_radial_zone']).count() / df[[
                'driver_id', 'pick_up_radial_zone', 'pickup_latitude'
            ]].groupby(['driver_id'])[['pickup_latitude']].count()
        temp = grouped_tmp.unstack(level=0).T
        temp.fillna(0, inplace=True)
        temp.reset_index(inplace=True)
        temp.drop(columns=['level_0'], inplace=True)
        pd.to_msgpack(cache_path, temp)
        print(f'Dumping to {cache_path}')
    return temp


def get_radial_coords(df):
    df['xpickup'], df['ypickup'], _, _ = utm.from_latlon(df['pickup_latitude'].values,
                                   df['pickup_longitude'].values)

    x_center, y_center, _, _ = utm.from_latlon(X_CENTER, Y_CENTER)
    print('center is ', x_center, y_center)
    df['x_diff'] = df.xpickup - x_center
    df['y_diff'] = df.ypickup - y_center
    df['r_radial'], df['theta_radial'] = cart2pol(df['x_diff'], df['y_diff'])


def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x) * 180 / np.pi
    return(r, theta)

def get_spatial_features_hex(df, resolution=6, use_cache=True):

    print('Now creating spatial features')
    cache_path = os.path.join(CACHE_DIR, f'hex_spatial_df.msgpack')
    if os.path.exists(cache_path) and use_cache:
        print(f'{cache_path} exists')
        temp = pd.read_msgpack(cache_path)
    else:
        minlat  = min(df.pickup_latitude)
        minlong = min(df.pickup_longitude)
        maxlat  = max(df.pickup_latitude)
        maxlong = max(df.pickup_longitude)
        geoJson = {'type': 'Polygon',
                   'coordinates': [[[minlat, minlong], [minlat, maxlong ], [maxlat, maxlong], [ maxlat, minlong ]]] }

        hexagons = list(h3.polyfill(geoJson, resolution))

        xy_pickup = utm.from_latlon(df.pickup_latitude.values,df.pickup_longitude.values)
        x_pickup = list(xy_pickup[0])
        y_pickup = list(xy_pickup[1])
        pickup_point = list(zip(x_pickup, y_pickup))

        poly_hex = dict()
        for i,hex in enumerate(hexagons):
            polygons = h3.h3_set_to_multi_polygon([hex], geo_json=False)
            a= np.array(polygons[0][0])
            b = utm.from_latlon(a[:,0],a[:,1])
            poly_hex[i] = list(zip(b[0], b[1]))

        pick_zone = np.zeros(len(df))-1
        for j,p in enumerate(pickup_point):
            point = Point(p)
            for i in range(len(poly_hex)):
                polygon = Polygon(poly_hex[i])
                if polygon.contains(point):
                    pick_zone[j] = int(i)
                    break

        df['pickup_zone'] = pick_zone

        grouped_tmp = df[['driver_id', 'pickup_zone', 'pickup_latitude']].groupby(
            ['driver_id', 'pickup_zone']).count() / df[[
                'driver_id', 'pickup_zone', 'pickup_latitude'
            ]].groupby(['driver_id'])[['pickup_latitude']].count()

        temp = grouped_tmp.unstack(level=0).T
        temp.fillna(0, inplace=True)
        temp.reset_index(inplace=True)
        temp.drop(columns=['level_0'], inplace=True)
        pd.to_msgpack(cache_path, temp)
        print(f'Dumping to {cache_path}')

    return temp


def create_modified_active_time(orders, use_cache=True, save_file=True, combine_pool=False):
    cache_path = os.path.join(CACHE_DIR, f'active_times_pool_{combine_pool}.msgpack')
    if os.path.exists(cache_path) and use_cache:
        print(f'{cache_path} exists')
        driver_stats_updated = pd.read_msgpack(cache_path)

    else:
        if combine_pool:
            orders = orders.copy()
            orders = group_pool(orders)

        driver_start_times = orders.loc[:, ['driver_id', 'ride_start_timestamp', 'ride_stop_timestamp', 'order_id']]\
            .drop_duplicates()
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

        driver_stats_updated = driver_stats_updated[cols]
        if save_file:
            pd.to_msgpack(cache_path, driver_stats_updated)
            print(f'Dumping to {cache_path}')

    return driver_stats_updated


def create_modified_active_time_through_decay(orders, use_cache=True, save_file=True, combine_pool=False):
    cache_path = os.path.join(CACHE_DIR, f'idle_times_old_pool_{combine_pool}.msgpack')
    if os.path.exists(cache_path) and use_cache:
        print(f'{cache_path} exists')
        driver_stats_updated = pd.read_msgpack(cache_path)

    else:
        if combine_pool:
            orders = orders.copy()
            orders = group_pool(orders)
        print("Creating the exponential decay")
        driver_start_times = orders.loc[:, ['driver_id', 'ride_start_timestamp', 'ride_stop_timestamp', 'order_id']]\
            .drop_duplicates()
        driver_start_times.sort_values(['driver_id', 'ride_start_timestamp'],
                                       inplace=True)
        driver_start_times['stop_time_shifted'] = driver_start_times.groupby(
            'driver_id')['ride_stop_timestamp'].shift(1)
        driver_start_times['diff'] = driver_start_times[
            'ride_start_timestamp'] - driver_start_times['stop_time_shifted']

        driver_start_times_no_na = driver_start_times.dropna()
        driver_start_times_no_na['diff'] = driver_start_times_no_na['diff'].dt.total_seconds() / 60

        mean_diff = driver_start_times_no_na['diff'].mean()
        tau = driver_start_times_no_na['diff'].median()

        lmbd = 1. / (mean_diff - tau)
        shape = 1. / lmbd

        size = driver_start_times_no_na.shape[0]
        driver_start_times_no_na['inactive_time'] = idle_time_est_old(driver_start_times_no_na['diff'], tau, shape, size=size)


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

        ##
        total_inactive_time = driver_start_times_no_na.groupby(
            'driver_id')['inactive_time'].sum().reset_index()
        ##
        driver_stats_updated = driver_stats.merge(
            total_inactive_time[['driver_id', 'inactive_time']],
            on='driver_id',
            how='left').fillna(0)
        driver_stats_updated['modified_active_time'] = driver_stats_updated[
            'active_time'] - driver_stats_updated['inactive_time']

        cols = [
            'driver_id', 'ride_duration', 'modified_active_time', 'inactive_time'
        ]

        driver_stats_updated = driver_stats_updated[cols]
        if save_file:
            pd.to_msgpack(cache_path, driver_stats_updated)
            print(f'Dumping to {cache_path}')

    return driver_stats_updated


def create_modified_active_time_through_decay2(orders, mult_factor, use_cache=True, save_file=True, combine_pool=False):
    cache_path = os.path.join(CACHE_DIR, f'idle_times_new_{mult_factor}_pool_{combine_pool}.msgpack')
    if os.path.exists(cache_path) and use_cache:
        print(f'{cache_path} exists')
        driver_stats_updated = pd.read_msgpack(cache_path)

    else:
        if combine_pool:
            orders = orders.copy()
            orders = group_pool(orders)
        print("Creating the exponential decay")
        driver_start_times = orders.loc[:, ['driver_id', 'ride_start_timestamp', 'ride_stop_timestamp', 'order_id']]\
            .drop_duplicates()

        driver_start_times.sort_values(['driver_id', 'ride_start_timestamp'],
                                       inplace=True)

        driver_start_times['start_time_shifted'] = driver_start_times.groupby(
            'driver_id')['ride_start_timestamp'].shift(-1)

        driver_start_times['diff'] = driver_start_times['start_time_shifted'] - driver_start_times['ride_stop_timestamp']

        driver_start_times_no_na = driver_start_times.dropna()

        driver_start_times_no_na['diff'] = driver_start_times_no_na['diff'].dt.total_seconds() / 60

        driver_start_times_no_na['hour'] = driver_start_times_no_na.ride_stop_timestamp.dt.hour

        tau = driver_start_times_no_na['diff'].median()

        driver_start_times_no_na['new_diff'] = np.maximum(0, (driver_start_times_no_na['diff'] - tau))

        stop_times = driver_start_times_no_na.ride_stop_timestamp
        driver_start_times_no_na['total_hour'] = (stop_times.dt.hour * 3600 + stop_times.dt.minute * 60 + stop_times.dt.second) / 3600

        # print(driver_start_times_no_na)

        # Modeling lambda
        stats_df = driver_start_times_no_na.groupby('hour')['new_diff'].agg(['count', 'mean']).reset_index()
        stats_df['count'] = stats_df['count'] / 10**7
        mean = np.mean(stats_df['count'])
        std = np.std(stats_df['count'])

        # Fit gaussian on counts
        from scipy.optimize import curve_fit
        from scipy import asarray as ar, exp

        x = list(stats_df.index)
        y = stats_df['count'].values

        n = len(x)  # the number of data
        mean = sum(x * y) / n  # note this correction
        sigma = sum(y * (x - mean) ** 2) / n  # note this correction

        def gaus(x, a, x0, sigma):
            return a * exp(-(x - x0) ** 2 / (2 * sigma ** 2))

        popt, pcov = curve_fit(gaus, x, y, p0=[1, mean, sigma])\

        # Find lambda for entire df
        driver_start_times_no_na['lmbd'] = gaus(driver_start_times_no_na['total_hour'], *popt)

        shape = 1. / (mult_factor*driver_start_times_no_na['lmbd'].values) #TODO revise

        driver_start_times_no_na['inactive_time'] = idle_time_est(driver_start_times_no_na['diff'], tau, shape)

        # print(driver_start_times_no_na[['ride_start_timestamp', 'ride_stop_timestamp', 'new_diff',
        #                                 'inactive_time', 'start_time_shifted']]
        #       [driver_start_times_no_na.driver_id == '0001860739024029fa3da2cad0ed4de2'])

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
        ##

        ##
        driver_ride_durations = orders.groupby('driver_id')[[
            'ride_duration'
        ]].sum().reset_index()
        ##

        # print("\n\n\n")

        # print(driver_ride_durations[driver_ride_durations.driver_id == '0001860739024029fa3da2cad0ed4de2'])

        ##
        # total driver active time
        driver_stats = driver_active_time[['driver_id', 'active_time']].merge(
            driver_ride_durations, on='driver_id', how='left')

        ##
        total_inactive_time = driver_start_times_no_na.groupby(
            'driver_id')['inactive_time'].sum().reset_index()
        ##
        driver_stats_updated = driver_stats.merge(
            total_inactive_time[['driver_id', 'inactive_time']],
            on='driver_id',
            how='left').fillna(0)
        driver_stats_updated['modified_active_time'] = driver_stats_updated[
            'active_time'] - driver_stats_updated['inactive_time']

        # print("\n\n\n")

        # print(driver_stats_updated[driver_stats_updated.driver_id == '0001860739024029fa3da2cad0ed4de2'])

        cols = [
            'driver_id', 'ride_duration', 'modified_active_time', 'inactive_time'
        ]

        driver_stats_updated = driver_stats_updated[cols]
        if save_file:
            pd.to_msgpack(cache_path, driver_stats_updated)
            print(f'Dumping to {cache_path}')

    return driver_stats_updated


def plot_active_time_with_k(dataframe, title):
    plt.title(f'{title}')
    dataframe[['modified_active_time']].plot.hist(bins=100)
    plt.show()


def create_radial_bins(df, cols=None):
    get_radial_coords(df)
    cuts = [NUM_CUTS_R_RADIAL, NUM_CUTS_THETA_RADIAL]
    for col, cut in zip(cols, cuts):
        print('checking bins name ', col, cut)
        # try:
        df[f'{col}_bin'] = pd.qcut(df[f'{col}'], cut).astype(str)
        # except Exception as e:
        #     import sys
        #     sys.exit()
        #     print(cut, col)
        print(df.head())
    df['pick_up_radial_zone'] = df['r_radial_bin'] + df['theta_radial_bin']