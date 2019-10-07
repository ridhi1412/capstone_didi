# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:11:27 2019

@author: rmahajan14
"""

import datetime
import pandas as pd
from utils import get_start_end_bins
from loader1 import read_data

date_str_list = ['20161128', '20161129', '20161130']

order = read_data('order', date='20161130', sample=1)
order.sort_values(by=['driver_id', 'ride_start_timestamp'], inplace=True)


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


#    return orders


def merge_order_df():
    """
    Concatenate order dataframes for given dates
    """

    df_new_list = []

    date_str_list = get_date_list()

    for date in date_str_list:
        order = read_data('order', date=date, sample=1)
        df_new_list += [order.copy()]

    orders = pd.concat(df_new_list, sort=False)

    return orders


def create_features():
    """
    Add all features
    """

    orders = merge_order_df()
    #    breakpoint()
    pool_rides(orders)
    get_start_end_bins(orders, ['ride_start_timestamp', 'ride_stop_timestamp'])

    df_new = orders.groupby(['driver_id']).agg({
        'ride_start_timestamp_bin': 'count',
        'ride_stop_timestamp_bin': 'count',
        'order_id': 'count',
        'is_pool': 'sum'
    }).reset_index()

    df_new.rename(
        columns={
            'order_id': 'num_total_rides',
            'is_pool': 'num_pool_rides'
        },
        inplace=True)

    df_new['% of pool rides'] = (
        df_new['num_pool_rides'] / df_new['num_total_rides'])

    return df_new


df_new = create_features()

#get_start_end_bins(df_new, date,
#                   ['ride_start_timestamp', 'ride_stop_timestamp'])

#    break
#    gps_df = read_data('gps', date=date, sample=1)
#    drivers = gps_df[['driver_id', 'order_id']].drop_duplicates()
#    orders = orders.merge(drivers, on='order_id', how='left')
#    df = orders.loc[orders['driver_id'] == '025a8a42a4cd1d0ca336d4743e98fe64']
#    df = orders.loc[orders['driver_id'] == '0009873b1084c284cc143db9d6cfdbf0']

#to experiment
#df['bin'] = pd.qcut(df['ride_start_timestamp'], 4)
#count = df['bin'].unique()
