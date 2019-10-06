# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:11:27 2019

@author: rmahajan14
"""


from utils import get_start_end_bins
from loader1 import read_data

#if __name__ == '__main__':
#    date = '20161128'
#    df = read_data('order', date=date, sample=1)
#    cols = ['ride_start_timestamp', 'ride_stop_timestamp']
#    get_start_end_bins(df, date, cols)

    #to experiment
    #df['bin'] = pd.qcut(df['ride_start_timestamp'], 4)
    #count = df['bin'].unique()

date = '20161128'
orders = read_data('order', date=date, sample=1)
orders['ride_stop_timestamp'].max()


gps_df = read_data('gps', date=date, sample=1)


drivers = gps_df[['driver_id', 'order_id']].drop_duplicates()
orders = orders.merge(drivers, on='order_id', how='left')



orders.sort_values(by=['driver_id', 'ride_start_timestamp'], inplace=True)
orders['shifted_end_time'] = orders['ride_stop_timestamp'].shift(1)
orders['shifted_driver'] = orders['driver_id'].shift(1)
orders['cond_1'] = orders['ride_start_timestamp'] < orders['shifted_end_time']
orders['cond_2'] = orders['shifted_driver'] == orders['driver_id']
orders['is_pool'] = (orders['cond_1'] & orders['cond_2'])

#df = orders.loc[orders['driver_id'] == '0009873b1084c284cc143db9d6cfdbf0']

df_new = orders.groupby(['driver_id']).agg(
        {'ride_start_timestamp':min,
         'ride_stop_timestamp':max,
          'order_id':'count',
          'is_pool': 'sum'}
        ).reset_index()

df_new.rename(columns={'order_id': 'num_total_rides',
               'is_pool': 'num_pool_rides'},
            inplace=True)

#df = orders.loc[orders['driver_id'] == '025a8a42a4cd1d0ca336d4743e98fe64']

df_new['% of pool rides'] = df_new['num_pool_rides']/df_new['num_total_rides']

get_start_end_bins(df_new, date, ['ride_start_timestamp','ride_stop_timestamp'])


