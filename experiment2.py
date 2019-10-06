# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:13:56 2019

@author: rmahajan14
"""
from loader1 import read_data


df_gps = read_data('gps', date='20161101', sample=1)
df_order = read_data('order', date='20161101', sample=1)