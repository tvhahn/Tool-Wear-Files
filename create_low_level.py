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
from datetime import datetime
import pickle
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

# location of all the split signals (these are all the pickles that were created in the create_split_data.py)
split_data_folder = Path(
    "interim_sample_data"
)
# split_data_folder = Path('/home/tim/Documents/Checkfluid-Project/notebooks/1.7-tvh-refactor-pipeline/_test')

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

# save as a csv
df_low.to_csv(("low_level_labels_TEST.csv"), index=False)
df_low.head()
