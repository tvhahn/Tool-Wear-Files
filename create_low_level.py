"""
Script that is used to create a labelled low_level dataframe and CSV.
The csv will include information of each individual split cut, along with 
the label (e.g. if it is failed or not).

"""

import scipy.io as sio
import numpy as np
import pandas as pd
import pickle
import os
import re
import matplotlib.pyplot as plt
import time
import sys
from datetime import datetime
import pickle
from zipfile import ZipFile
import zipfile
import zlib
from pathlib import Path
from load_data import (
    load_cut_files,
    high_level_csv,
    split_df,
    check_date,
    extract_data_csv,
    low_level_df,
)
from feature_engineering import (
    feat_min_value,
    feat_max_value,
    feat_rms_value,
    feat_std_value,
    feat_kurtosis,
    feat_freq_pk_s1,
    feat_freq_pk_s1_norm,
    feat_freq_mean,
    feat_freq_std,
)

# define features that you want calculated on low_level_df
# format:
    # {"dictionary_key": [feature_function, "feature_name"]}

features = {"min_current_main":[feat_min_value, "current_main", 'spindle_main'],
            "max_current_main":[feat_max_value, "current_main", 'spindle_main'],
            "min_current_sub":[feat_min_value, "current_sub", 'spindle_sub'],
            "max_current_sub":[feat_max_value, "current_sub", 'spindle_sub'],
            "rms_current_main":[feat_rms_value, "current_main", 'spindle_main'],
            "rms_current_sub":[feat_rms_value, "current_sub", 'spindle_sub'],
            "std_current_main": [feat_std_value, "current_main", 'spindle_main'], 
            "std_current_sub": [feat_std_value, "current_sub", 'spindle_sub'],       
            "kur_current_main":[feat_kurtosis, "current_main", 'spindle_main'],
            "kur_current_sub":[feat_kurtosis, "current_sub", 'spindle_sub'],
            "freq_pks1_current_main": [feat_freq_pk_s1, "current_main", 'spindle_main'], 
            "freq_pks1_current_sub": [feat_freq_pk_s1, "current_sub", 'spindle_sub'],
            "freq_pks1_norm_current_main":[feat_freq_pk_s1_norm, "current_main", 'spindle_main'],
            "freq_pks1_norm_current_sub":[feat_freq_pk_s1_norm, "current_sub", 'spindle_sub'],
            "freq_mean_current_main": [feat_freq_mean, "current_main", 'spindle_main'], 
            "freq_mean_current_sub": [feat_freq_mean, "current_sub", 'spindle_sub'],
            "freq_std_current_main": [feat_freq_std, "current_main", 'spindle_main'], 
            "freq_std_current_sub": [feat_freq_std, "current_sub", 'spindle_sub'],
           }

# location of the high_level csv that has been labelled with faile/not-failed labels
high_level_label_location = Path(
    "high_level_LABELLED.csv"
)

# location of the zip folders containing the split pickles
zip_path = Path('/home/tvhahn/projects/def-mechefsk/tvhahn/split_data_stable_speed_no_pad_ind_2020.01.21_ZIP')

# setup the location where the split cut data will be stored.
# folder location will be created if does not already exist
Path("/home/tvhahn/scratch/interim_data").mkdir(parents=True, exist_ok=True)
scratch_path = Path("/home/tvhahn/scratch/interim_data")

file_name = sys.argv[1]
file_folder_index = file_name.split(sep='.')[0]

# extract zip file
with ZipFile(zip_path / file_name,'r') as zip_file:
    # setup the location where the split cut data will be stored.
    # folder location will be created if does not already exist
    zip_file.extractall(path=(scratch_path / file_folder_index))

# location of all the split signals (these are all the pickles that were created in the create_split_data.py)
split_data_folder = scratch_path / file_folder_index 


# read the high_level csv
df1 = pd.read_csv(high_level_label_location)
df1 = df1.dropna(subset=["failed"])  # drop rows that do not have a failed indicator

# Create the low-level df
# we will be calculating singlar values for the signals as well
df2 = low_level_df(
    split_data_folder,
    features,
    svd_feature=True,
    fft_features=True,
    list_of_svd_signals=["current_sub", "current_main"],
    svd_feat_count=25,
    svd_window_size=100,
)

# label the individual cuts in the low-level df as failed or not
df_low = check_date(df1, df2)

# create a save folder for the CSVs
Path("temp_csv").mkdir(parents=True, exist_ok=True)


name_of_csv = "temp_csv/low_level_labels_{}.csv".format(str(file_folder_index))

# save as a csv
df_low.to_csv((name_of_csv), index=False)
df_low.head()
print('Created file: ', name_of_csv)
