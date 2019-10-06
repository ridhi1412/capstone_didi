# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:26:39 2019

@author: rmahajan14
"""
import argparse

parser = argparse.ArgumentParser(description="Transfer bep reports to BCS")
parser.add_argument(
    "-d",
    "--data_dir",
    required=True,
    help="Data Directory",
)
parser.add_argument(
    "-c",
    "--cache_dir",
    required=True,
    help="Cache Directory",
)
args = vars(parser.parse_args())
DATA_DIR = args["data_dir"]
CACHE_DIR = args["cache_dir"]


# e.g. -d r"C:\Users\rmahajan14\capstone_data\data" -c r"C:\Users\rmahajan14\capstone_data\cache"