# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:11:27 2019

@author: rmahajan14
"""

import os
import datetime
import pandas as pd
from utils import get_start_end_bins, get_spatial_features, create_modified_active_time
from loader1 import read_data
from common import CACHE_DIR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor

#date_str_list = ['20161128', '20161129', '20161130']
#order = read_data('order', date='20161130', sample=1)
#order.sort_values(by=['driver_id', 'ride_start_timestamp'], inplace=True)


def get_date_list(start='2016-11-01', end='2016-11-30'):
    """
    Get dates to merge orders dataframe for
    """

    date_str_list = [
        datetime.datetime.strftime(date, '%Y%m%d')
        for date in pd.date_range(start, end)
    ]

    return date_str_list


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



def merge_order_df(start='2016-11-01', end='2016-11-30',
                   use_cache=True):
    """
    Concatenate order dataframes for given dates
    """

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
        pd.to_msgpack(cache_path, orders)
        print(f'Dumping to {cache_path}')
    return orders


def create_features(start='2016-11-01', end='2016-11-30', use_cache=True):
    """
    Add all features
    """

    cache_path = os.path.join(CACHE_DIR, f'features_orders.msgpack')
    if os.path.exists(cache_path) and use_cache:
        print(f'{cache_path} exists')
        df_new = pd.read_msgpack(cache_path)
    else:
        orders = merge_order_df(start, end)
        pool_rides(orders)
        get_start_end_bins(orders,
                           ['ride_start_timestamp', 'ride_stop_timestamp'])

        #        breakpoint()

        print('a')
        grouped_tmp = orders[[
            'driver_id', 'ride_start_timestamp_bin', 'order_id'
        ]].groupby(['driver_id', 'ride_start_timestamp_bin'
                    ]).count() / orders[[
                        'driver_id', 'ride_start_timestamp_bin', 'order_id'
                    ]].groupby(['driver_id'])[['order_id']].count()

        temp1 = grouped_tmp.unstack(level=0)
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
        pd.to_msgpack(cache_path, df_new)
        print(f'Dumping to {cache_path}')
#        breakpoint()
        df_final = pd.merge(df_new, temp1, on=['driver_id'], how='inner')

    return df_final

def get_final_df_reg(use_cache=True):
    cache_path = os.path.join(CACHE_DIR, f'final_df_reg.msgpack')
    if os.path.exists(cache_path) and use_cache:
        print(f'{cache_path} exists')
        df_final = pd.read_msgpack(cache_path)
    else:
        start = '2016-11-01'
        end = '2016-11-30'
        orders = merge_order_df(start=start, end=end)
        print('orders')
        
        target_df = create_modified_active_time(orders)
        target_df['target'] = target_df['ride_duration'] / target_df[
            'modified_active_time_with_rules']
        target_df.sort_values('driver_id', inplace=True)
        
        print('1e')
        df_final = create_features(
            start='2016-11-01', end='2016-11-30', use_cache=False)
        print('1f')
        spatial_df = get_spatial_features(orders)
        print('spatial')
        
        df_final = pd.merge(df_final, spatial_df, on=['driver_id'], how='inner')
        df_final.sort_values('driver_id', inplace=True)
        df_final.set_index('driver_id', inplace=True)
        pd.to_msgpack(cache_path, df_final)
    return df_final


df_final = get_final_df_reg()

##X = df_final.drop(columns=['num_total_rides'])
#X = df_final
#
#xtrain, xtest, ytrain, ytest = train_test_split(X, target_df['target'])
#
#sc = StandardScaler()
#xtrain_sc = sc.fit_transform(xtrain)
#
#rr = RandomForestRegressor()
#rr.fit(xtrain_sc, ytrain)
#rr.fit(xtrain_sc, ytrain)
#
##print(rr.coef_)
#print(rr.score(xtrain_sc, ytrain))

#TODO think
#temp_in = temp.reset_index()
#temp_in = temp_in.drop(columns=['level_0'])
#temp_str = temp_in.drop_duplicates()
#len(list(temp_in['driver_id'].unique()))
#aa= temp_str.loc[temp_str['driver_id'] == '0000131d486b69eb77ab6e9e7cca9f4c'].T

#    get_start_end_bins(df_new, date,
#                   ['ride_start_timestamp', 'ride_stop_timestamp'])
#    break
#    gps_df = read_data('gps', date=date, sample=1)
#    drivers = gps_df[['driver_id', 'order_id']].drop_duplicates()
#    orders = orders.merge(drivers, on='order_id', how='left')
#    df = orders.loc[orders['driver_id'] == '025a8a42a4cd1d0ca336d4743e98fe64']
#    df = orders.loc[orders['driver_id'] == '0009873b1084c284cc143db9d6cfdbf0']
