# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:28:24 2019

@author: rmahajan14
"""

import os
import sys
import tarfile
import pandas as pd
from datetime import datetime
from common import DATA_DIR, CACHE_DIR


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
        #        df[col] = df[col].apply(lambda x: convert(x))
        #        df[col] = pd.DatetimeIndex(pd.to_datetime(df[col])).tz_localize('UTC')\
        #            .tz_convert('Asia/Shanghai').tz_localize(None)

        df[col] = pd.to_datetime(df[col], unit='s')
        df[col] = pd.DatetimeIndex(df[col]).tz_localize('UTC')\
            .tz_convert('Asia/Shanghai').tz_localize(None)


def ride_duration(df):
    """
    Add column for duration of ride in minutes
    :param df: Input dataframe
    :return:
    """
    assert 'ride_start_timestamp' in list(
        df.columns) and 'ride_stop_timestamp' in list(df.columns)

    df['ride_duration'] = (df.ride_stop_timestamp -
                           df.ride_start_timestamp).dt.total_seconds() / 60

    return df


def load_all(use_cache=True, override=False):
    """
    Read in compressed files and cache in pandas readible files locally.
    Needed to be run on the first ever run.
    :param use_cache:
    :param override: Override directory check and run this function
    :return:
    """
    # Check if files already exist in CACHE_DIR and warn user
    if not os.path.isdir(CACHE_DIR):
        os.mkdir(CACHE_DIR)

    if len(os.listdir(CACHE_DIR)) > 1 and not override:
        print(
            "Some files already exist in your CACHE_DIR. If you still want to run this function,\
              run with override=True")
        return

    i = 1
    for file in os.listdir(DATA_DIR):
        print(f'Processing {i} of 30 files')
        file_path = os.path.join(DATA_DIR, file)
        tar = tarfile.open(file_path, "r:gz")
        for member in tar.getmembers():
            cache_path = os.path.join(CACHE_DIR, f'{member.name}.msgpack')
            print(member.name)
            if member.name.startswith('order'):
                date_equality = pd.to_datetime(member.name[6:]).date()
            if os.path.exists(cache_path):
                print(f'{cache_path} exists')
#                continue
            if member.name.startswith('gps'):
                col_names = [
                    'driver_id', 'order_id', 'timestamp', 'longitude',
                    'latitude'
                ]
                timecols = ['timestamp']
            elif member.name.startswith('order'):
                col_names = [
                    'order_id', 'ride_start_timestamp', 'ride_stop_timestamp',
                    'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                    'dropoff_latitude'
                ]
                timecols = ['ride_start_timestamp', 'ride_stop_timestamp']
            else:
                sys.exit()
            f = tar.extractfile(member)
            if f is not None:
                df = pd.read_csv(f, header=None, names=col_names)
                df.drop_duplicates(inplace=True)
                convert_unix_ts(df, timecols)
                if member.name.startswith('gps'):
                    df.sort_values(by=['driver_id', 'timestamp'], inplace=True)
                    df_gps_date = pd.to_datetime(df.head(n=1)['timestamp'].iloc[0]).date()
                    df_gps = df[['driver_id', 'order_id']].drop_duplicates()
                if member.name.startswith('order'):
                    ride_duration(df)
                    df.sort_values(
                        by=['order_id', 'ride_start_timestamp'], inplace=True)
                    assert(date_equality == df_gps_date)
#                    TODO
#assert(date_equality == df_gps_date)
#                    breakpoint()
                    df = df.merge(df_gps, on='order_id', how='left')

                pd.to_msgpack(cache_path, df)
        i += 1

def load_processed_dfs():
    pass





# todo incorporate multi-date file reads?
def read_data(data_type, date='20161101', sample=1):
    """
    Reads in data from cached messagepacks stored locally
    :param data_type: 'gps' or 'order'
    :param date: date string without space - yyyymmdd. Limited to single date as each file too big
    :param sample: Fraction of randomly sampled data (without replacement) to return. Defaults to full data
    :return: pandas dataframe of requested file
    """
    file_name = data_type + '_' + date
    file_path = os.path.join(CACHE_DIR, f'{file_name}.msgpack')
    print(f'file path is {file_path}')
    df = pd.read_msgpack(file_path)
    if sample < 1:
        df_sample = df.sample(frac=sample, random_state=42)
    elif sample == 1:
        df_sample = df
    else:
        sys.exit('Sample size can be atmost 1')

    return df_sample


if __name__ == '__main__':
    load_all(override=True)
