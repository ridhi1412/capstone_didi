# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:28:24 2019

@author: aksmit94
"""

import pandas as pd
import utm
import os

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
        
    temp_pickup = df.apply(lambda x: utm.from_latlon(x['pickup_latitude'], x['pickup_longitude'])[0:2], axis=1)
    temp_drop = df.apply(lambda x: utm.from_latlon(x['dropoff_latitude'], x['dropoff_longitude'])[0:2], axis=1)

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
    grouped_tmp = df[['driver_id', 'pick_up_zone', 'pickup_latitude']].groupby([
            'driver_id', 'pick_up_zone']).count(
            ) / df[['driver_id', 'pick_up_zone', 'pickup_latitude']].groupby(
        ['driver_id'])[['pickup_latitude']].count()
#    breakpoint()
#    grouped_tmp = grouped_tmp[['dropoff_latitude']]
    
#    df1 = grouped_tmp.unstack(level=1)
    temp = grouped_tmp.unstack(level=0).T
#    df1m = grouped_tmp.unstack(level=-1)
    
#    (df.groupby(['driver_id', 'pick_up_zone']).count() / df.groupby(
#        ['driver_id']).count())

    temp.fillna(0, inplace=True)
    return temp

