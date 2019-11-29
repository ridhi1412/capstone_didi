#%%
from loader1 import *
from utils import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
gps = read_data('gps', sample=0.05)
#sys.argv = ["_", "/Users/ashwin/Documents/Courses/Capstone/Capstone - Data", "/Users/ashwin/Documents/Courses/Capstone//Ride_Data_Cache/"]

#%%
print(sys.argv)
#%%
gps = read_data('gps', sample=0.000005).filter(10)
orders = read_data('order', sample=0.000005).head(10)
print('Read Data')

print(gps.columns)
print(orders.columns)

print(list(gps.head(2)))
print(list(orders.head(2)))
# #%%
# gps_updated = convert_unix_ts(gps, timecols=['timestamp'])
# orders_updated = convert_unix_ts(orders, timecols=['ride_start_timestamp', 'ride_stop_timestamp'])
# orders_updated = ride_duration(orders_updated)
#
# print('Orders Data')
#
# #%%
# print("No of duplicate rows in Orders is  ", orders.shape[0] - orders.drop_duplicates().shape[0])
# print(orders.shape[0])
#
# #%%
# print("No of duplicate rows in GPS is  ", gps.shape[0] - gps.drop_duplicates().shape[0])
# print(gps.shape[0])
#
# #%%
# drivers = gps_updated[['driver_id', 'order_id']].drop_duplicates()
# orders_updated = orders_updated.merge(drivers, on='order_id', how='left')
#
# driver_ride_durations = orders_updated.groupby('driver_id')[['ride_duration']].sum().reset_index()
#
# print('Ride Durations')
#
# #%%
#
# # The drivers who have only 1 ride
# print(orders_updated.columns)
# driver_ride_count = orders_updated.groupby('driver_id').size().reset_index(name='Count')
# print("Drivers with 1 ride only are ", driver_ride_count[driver_ride_count.Count == 1].shape[0])
# # 5285
#
# #%%
# driver_day_min = pd.DataFrame(orders_updated.groupby('driver_id')['ride_start_timestamp'].min()).reset_index()
# driver_day_max = pd.DataFrame(orders_updated.groupby('driver_id')['ride_stop_timestamp'].max()).reset_index()
# driver_active_time = driver_day_min.merge(driver_day_max, on='driver_id', how='left')
# driver_active_time['active_time'] = (driver_active_time['ride_stop_timestamp'] - driver_active_time['ride_start_timestamp']).dt.total_seconds() / 60
#
# print('Min and Max')
#
# #%%
# # Drivers with short active time ie
#
# # 1 quantile
# print(driver_active_time['active_time'].quantile(0.1))
# print(driver_active_time.describe())
#
# #
#
#
# #%%
# # total driver active time
# driver_stats = driver_active_time[['driver_id', 'active_time']].merge(driver_ride_durations, on='driver_id', how='left')
#
# # Filtering possible bad rows
# # driver_stats = driver_stats[driver_stats.active_time > driver_stats.ride_duration]
#
# #%%
# print(driver_stats[driver_stats.active_time < driver_stats.ride_duration].shape)
#
# #%%
# plt.figure(figsize=(10,10))
# plt.scatter(driver_stats['active_time'], driver_stats['ride_duration'], c='b', alpha=0.2)
# plt.ylabel('total ride duration')
# plt.xlabel('active time')
# plt.plot()
# plt.show()
#
# #%%
# gps.columns
