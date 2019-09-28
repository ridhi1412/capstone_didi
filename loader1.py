# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:28:24 2019

@author: rmahajan14
"""

import os
import tarfile
import pandas as pd
from common import DATA_DIR, CACHE_DIR
from sys import exit


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

    if len(os.listdir(CACHE_DIR)) > 1:
        print("Some files already exist in your CACHE_DIR. If you still want to run this function,\
              run with override=True")
        return

    i = 1
    for file in os.listdir(DATA_DIR):
        print('Processing {} of 30 files'.format(i))
        file_path = os.path.join(DATA_DIR, file)
        tar = tarfile.open(file_path, "r:gz")
        for member in tar.getmembers():
            cache_path = os.path.join(CACHE_DIR, f'{member.name}.msgpack')
            print(member.name)
            if member.name.startswith('gps'):
                col_names = ['driver_id', 'order_id', 'timestamp', 'longitude', 'latitude']
            else:
                col_names = ['order_id', 'ride_start_timestamp', 'ride_stop_timestamp', 'pickup_longitude',
                             'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
            f = tar.extractfile(member)
            if f is not None:
                df = pd.read_csv(f, header=None, names=col_names)
                pd.to_msgpack(cache_path, df)
        i += 1


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
    df = pd.read_msgpack(file_path)
    if sample < 1:
        df_sample = df.sample(frac=sample, random_state=42)
    else:
        df_sample = df

    return df_sample

# load_all()


