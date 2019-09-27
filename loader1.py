# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:28:24 2019

@author: rmahajan14
"""

import os
import tarfile
import pandas as pd
from common import DATA_DIR, CACHE_DIR

#df_path = os.path.join(DATA_DIR, '2e65b3139cf147429f5b45b6b33e8807.tar.gz')



def load_all(use_cache=True):
#    cache_path = my_path_join(CACHE_PATH, f'{buy_or_sell}_reader.msgpack')
    for file in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file)
        tar = tarfile.open(file_path, "r:gz")
        for member in tar.getmembers():
             cache_path = os.path.join(CACHE_DIR, f'{member.name}.msgpack')
             print(member.name)
             f = tar.extractfile(member)
             if f is not None:
                 df = pd.read_csv(f, header=None)
                 pd.to_msgpack(cache_path, df)


load_all()

#df0 = df.iloc[0]
#df1 = df.iloc[1:nan_index]
#df2 = df.iloc[nan_index+1:]
#
#df1_ = df1[0].str.split(',', expand=True)
##df2_ = df2[0].str.split(',', expand=True)
#
#a = df1.head(n=1000)
#a = a.iloc[1:]
#df1_ = a[0].str.split(',', expand=True)

#import tarfile
#tar = tarfile.open(df_path, "r:gz")
#for member in tar.getmembers():
#     print(member.name)
##     df = pd.read_csv(df_path, sep='\t', compression='gzip', header=None)
#     f = tar.extractfile(member)
#     if f is not None:
#         pass
#         content = pd.read_csv(f, header=None)
#         break