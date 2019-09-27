# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:28:24 2019

@author: rmahajan14
"""

import os
import tarfile
import pandas as pd
from common import DATA_DIR, CACHE_DIR


def load_all(use_cache=True):
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

