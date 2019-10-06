# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:11:27 2019

@author: rmahajan14
"""


from utils import get_start_end_bins
from loader1 import read_data

if __name__ == '__main__':
    date = '20161128'
    df = read_data('order', date=date, sample=1)
    cols = ['ride_start_timestamp', 'ride_stop_timestamp']
    get_start_end_bins(df, date, cols)

    #to experiment
    #df['bin'] = pd.qcut(df['ride_start_timestamp'], 4)
    #count = df['bin'].unique()







