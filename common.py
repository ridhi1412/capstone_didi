# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:26:39 2019

@author: rmahajan14
"""
import sys
import os

path = [r'../Capstone - Data', r'../Ride_Data_Cache']

if os.path.isdir(path[0]):
    DATA_DIR = path[0]
    CACHE_DIR = path[1]
else:
    DATA_DIR = sys.argv[1]
    CACHE_DIR = sys.argv[2]

# "P:\rmahajan14\capstone_data\data" "P:\rmahajan14\capstone_data\cache"

#import argparse

#parser = argparse.ArgumentParser(description="Transfer bep reports to BCS")
#parser.add_argument(
#    "-d",
#    "--data_dir",
#    required=True,
#    help="Data Directory",
#)
#parser.add_argument(
#    "-c",
#    "--cache_dir",
#    required=True,
#    help="Cache Directory",
#)
#args = vars(parser.parse_args())
#DATA_DIR = args["data_dir"]
#CACHE_DIR = args["cache_dir"]
