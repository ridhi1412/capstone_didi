# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:11:27 2019

@author: rmahajan14
"""

import sys
import os
import datetime
import pandas as pd
import numpy as np
from utils import (get_start_end_bins, get_spatial_features, get_spatial_features_hex,
                   create_modified_active_time, create_modified_active_time_through_decay,
                   create_modified_active_time_through_decay2, get_spatial_features_radial, get_surv_prob, pool_rides)
from loader1 import read_data
from common import CACHE_DIR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import time



def get_date_list(start='2016-11-01', end='2016-11-30'):
    """
    Get dates to merge orders dataframe for
    """

    date_str_list = [
        datetime.datetime.strftime(date, '%Y%m%d')
        for date in pd.date_range(start, end)
    ]

    return date_str_list






def merge_order_df(start='2016-11-01', end='2016-11-30',
                   use_cache=True, remove_pool=False):
    """
    Concatenate order dataframes for given dates
    """
    if remove_pool:
        cache_path = os.path.join(CACHE_DIR, f'merged_orders_no_pool.msgpack')
    else:
        cache_path = os.path.join(CACHE_DIR, f'merged_orders.msgpack')
    if os.path.exists(cache_path) and use_cache:
        print(f'{cache_path} exists')
        orders = pd.read_msgpack(cache_path)
    else:

        df_new_list = []

        date_str_list = get_date_list(start=start, end=end)

        for date in date_str_list:
            order = read_data('order', date=date, sample=1)
            df_new_list += [order.copy()]

        orders = pd.concat(df_new_list, sort=False)
        ##################################
        # Removing orders where the ride duration is greater than 180 minutes
        orders = orders[orders.ride_duration <= 180]
        ##################################
        pd.to_msgpack(cache_path, orders)
        print(f'Dumping to {cache_path}')
    return orders


def unstack_func(grouped_df):
    """
    Transposes grouped table by making groups as columns
    :param grouped_df: pandas grouped dataframe
    :return: pandas dataframe with groups as columns
    """

    temp1 = grouped_df.unstack(level=0)
    temp1.fillna(0, inplace=True)
    #        breakpoint()
    temp1.reset_index(inplace=True)
    temp1 = temp1.T
    temp1.reset_index(inplace=True)

    cols = temp1.iloc[0]
    temp1 = temp1.loc[1:]
    temp1.columns = cols

    print('b')

    temp1.rename(columns={'': 'driver_id'}, inplace=True)
    temp1.drop(columns=['ride_start_timestamp_bin'], inplace=True)
    new_cols = [temp1.columns[0]] + [str(x) for x in temp1.columns[1:]]
    temp1.columns = new_cols
    return temp1


def groupby_1_count(orders, use_cache=True, use_radial=False):
    """
    Grouping to get number of rides of a driver in each time bin
    :param orders: Orders dataframe with time bin columns
    :param use_cache: Use previous cache or not
    :return: Grouped pandas dataframe
    """
    if use_radial:
        group_col = 'pick_up_radial_zone'
    else:
        group_col = 'ride_start_timestamp_bin'

    cache_path = os.path.join(CACHE_DIR, f'groupby1.msgpack')
    if use_cache and os.path.exists(cache_path):
        temp1 = pd.read_msgpack(cache_path)
        print(f'Loading from {cache_path}')
    else:
        grouped_tmp = orders[[
            'driver_id', 'ride_start_timestamp_bin', 'order_id'
        ]].groupby(['driver_id', 'ride_start_timestamp_bin'
                    ]).count() / orders[[
                        'driver_id', 'ride_start_timestamp_bin', 'order_id'
                    ]].groupby(['driver_id'])[['order_id']].count()
        temp1 = unstack_func(grouped_tmp)
        pd.to_msgpack(cache_path, temp1)
        print(f'Dumping to {cache_path}')
    return temp1


def groupby_2_sum(orders, use_cache=True):
    """
    Grouping to get sum of ride durations of a driver in each time bin
    :param orders: Orders dataframe with time bin columns
    :param use_cache: Use previous cache or not
    :return: Grouped pandas dataframe
    """
    cache_path = os.path.join(CACHE_DIR, f'groupby2.msgpack')
    if use_cache and os.path.exists(cache_path):
        temp2 = pd.read_msgpack(cache_path)
        print(f'Loading from {cache_path}')
    else:
        grouped_tmp_perc_active = orders[[
            'driver_id', 'ride_start_timestamp_bin', 'ride_duration'
        ]].groupby(['driver_id', 'ride_start_timestamp_bin'
                    ])[['ride_duration']].sum() / orders[[
                        'driver_id', 'ride_start_timestamp_bin', 'ride_duration'
                    ]].groupby(['driver_id'])[['ride_duration']].sum()
        temp2 = unstack_func(grouped_tmp_perc_active)
        pd.to_msgpack(cache_path, temp2)
        print(f'Dumping to {cache_path}')
    return temp2


def create_features(start='2016-11-01', end='2016-11-30', use_cache=True):
    """
    Creates all features going into the linear model
    :param start: Start date of rides
    :param end: End date of rides
    :param use_cache: Use existing cache
    :return: Returns df with all features
    """

    cache_path = os.path.join(CACHE_DIR, f'features_orders.msgpack')
    if os.path.exists(cache_path) and use_cache:
        print(f'{cache_path} exists')
        df_final = pd.read_msgpack(cache_path)
    else:
        orders = merge_order_df(start, end, use_cache=True)
        pool_rides(orders)
        get_start_end_bins(orders,
                           ['ride_start_timestamp', 'ride_stop_timestamp'])

        #        breakpoint()

        print('a')
        import time
        a = time.time()
        temp1 = groupby_1_count(orders, use_cache=True)

        temp2 = groupby_2_sum(orders, use_cache=True)

        print(time.time() - a)

        df_new = orders.groupby(['driver_id']).agg({
            'order_id': 'count',
            'is_pool': 'sum'
        }).reset_index()

        print('c')
        df_new.rename(
            columns={
                'order_id': 'num_total_rides',
                'is_pool': 'num_pool_rides'
            },
            inplace=True)

        df_new['% of pool rides'] = (
            df_new['num_pool_rides'] / df_new['num_total_rides'])
        print('d')

        print(f'Dumping to {cache_path}')

        df_final = pd.merge(df_new, temp1, on=['driver_id'], how='inner')

        # TODO check

        df_final = pd.merge(df_final, temp2, on=['driver_id'], how='inner', suffixes=('_count', '_sum'))
        pd.to_msgpack(cache_path, df_final)
    return df_final


def get_final_df_reg(use_cache=False, decay='New Decay', mult_factor=1,
                     add_idle_time=False, use_radial_features=False, spatial_type='grid', combine_pool=False):
    cache_path = os.path.join(CACHE_DIR, f'final_df_reg.msgpack')
    cache_path_idle_time = os.path.join(CACHE_DIR, f'idle_times.msgpack')
    # if os.path.exists(cache_path) and os.path.exists(cache_path_idle_time) and use_cache:
    if False:
        print(f'{cache_path} exists')
        print(f'{cache_path_idle_time} exists')
        df_final = pd.read_msgpack(cache_path)
        target_df = pd.read_msgpack(cache_path_idle_time)
    else:
        start = '2016-11-01'
        end = '2016-11-30'
        orders = merge_order_df(start=start, end=end, use_cache=True)
        print('orders')

        t1 = time.time()
        print('Decay Calculation')
        if decay == 'No Decay':
            print("No Decay")
            target_df = create_modified_active_time(orders, use_cache=False, combine_pool=combine_pool)
            target_df['target'] = target_df['ride_duration'] / target_df[
                'modified_active_time_with_rules']
            target_df.sort_values('driver_id', inplace=True)
        elif decay == 'Old Decay':
            print("Old Decay")
            target_df = create_modified_active_time_through_decay(orders, use_cache=False, combine_pool=combine_pool)
            target_df['target'] = target_df['ride_duration'] / target_df[
                'modified_active_time']
            target_df.sort_values('driver_id', inplace=True)
        elif decay == 'New Decay':
            print("New Decay")
            target_df = create_modified_active_time_through_decay2(orders, mult_factor=mult_factor,
                                                                   use_cache=False, combine_pool=combine_pool)
            target_df['target'] = target_df['ride_duration'] / target_df[
                'modified_active_time']
            target_df.sort_values('driver_id', inplace=True)
        elif decay == 'Survival':
            print("Survival")
            target_df = get_surv_prob(orders, use_cache=False, combine_pool=combine_pool)
            target_df['target'] = target_df['ride_duration'] / target_df[
                'survival_active_time']
            target_df.sort_values('driver_id', inplace=True)
        else:
            raise NotImplementedError('Decay can only take 4 values')

        print(f"Decay Calculation done in {time.time() - t1}")

        print('1e')
        t1 = time.time()
        df_final = create_features(
            start='2016-11-01', end='2016-11-30', use_cache=True)
        # TODO change to True

        print(f"Features created in {time.time() - t1}")
        t1 = time.time()
        print('1f')
        if spatial_type.lower() == 'radial':
            spatial_df = get_spatial_features_radial(orders, use_cache=True)
        elif spatial_type.lower() == 'grid':
            spatial_df = get_spatial_features(orders, use_cache=True)
        elif spatial_type.lower() == 'hex':
            spatial_df = get_spatial_features_hex(orders, use_cache=True)
        elif spatial_type.lower() == 'both':
            spatial_df_radial = get_spatial_features_radial(orders, use_cache=True)
            spatial_df_grid = get_spatial_features(orders)
            spatial_df = pd.merge(spatial_df_grid, spatial_df_radial, on='driver_id')
        else:
            raise NotImplementedError()

        print('spatial')
        print(f"Spatial Calculation done in {time.time() - t1}")

        df_final = pd.merge(df_final, spatial_df, on=['driver_id'], how='inner')

        ##################################################
        ### Adding inactive time as a feature
        if decay != 'No Decay':
            df_final = pd.merge(df_final, target_df[['driver_id', 'inactive_time']], on=['driver_id'], how='inner')
        ##################################################

        df_final.sort_values('driver_id', inplace=True)
        df_final.set_index('driver_id', inplace=True)
        if sum(df_final.index != target_df['driver_id']) > 0:
            print("Index must match")
            sys.exit(0)
        pd.to_msgpack(cache_path, df_final)
    return df_final, target_df


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


if __name__ == '__main__':
    df_final, target_df = get_final_df_reg(use_cache=False)

# #X = df_final.drop(columns=['num_total_rides'])
# X = df_final
#
# xtrain, xtest, ytrain, ytest = train_test_split(X, target_df['target'])
#
# sc = StandardScaler()
# xtrain_sc = sc.fit_transform(xtrain)
#
# rr = RandomForestRegressor()
# rr.fit(xtrain_sc, ytrain)
# rr.fit(xtrain_sc, ytrain)
#
# #print(rr.coef_)
# print(rr.score(xtrain_sc, ytrain))

# TODO think
# temp_in = temp.reset_index()
# temp_in = temp_in.drop(columns=['level_0'])
# temp_str = temp_in.drop_duplicates()
# len(list(temp_in['driver_id'].unique()))
# aa= temp_str.loc[temp_str['driver_id'] == '0000131d486b69eb77ab6e9e7cca9f4c'].T

#    get_start_end_bins(df_new, date,
#                   ['ride_start_timestamp', 'ride_stop_timestamp'])
#    break
#    gps_df = read_data('gps', date=date, sample=1)
#    drivers = gps_df[['driver_id', 'order_id']].drop_duplicates()
#    orders = orders.merge(drivers, on='order_id', how='left')
#    df = orders.loc[orders['driver_id'] == '025a8a42a4cd1d0ca336d4743e98fe64']
#    df = orders.loc[orders['driver_id'] == '0009873b1084c284cc143db9d6cfdbf0']
