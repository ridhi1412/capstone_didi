# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:28:24 2019

@author: rmahajan14
"""

import os
import pandas as pd
from common import DATA_DIR

df_path = os.path.join(DATA_DIR, '2e65b3139cf147429f5b45b6b33e8807.tar.gz')
df = pd.read_csv(df_path, skiprows=1, compression='gzip', header=None, error_bad_lines=False)
df_index = list(df.index)

df2 = pd.read_table(df_path, compression='gzip', header=None)

df_not_index = list((set(list(range(len(df2))))).difference(set(df_index)))

#
#
#df2 = 

df2 = pd.read_table(df_path, compression='gzip', header=None)
#aaa = df2.iloc[39829910:39829920]
#bbb = aaa.isnull()
#
#xxx = df2.isnull()
#
#xxx = xxx.loc[xxx[0] == True]
#aaa = df2.iloc[39829910:39829920]