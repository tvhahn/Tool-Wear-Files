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
    svd_columns,
    feat_svd_values,
)

# create the dictionary of all the features that are to be
# included in the final csv
features = {
    "min_current_main": [feat_min_value, "current_main"],
    "max_current_main": [feat_max_value, "current_main"],
    "min_current_sub": [feat_min_value, "current_sub"],
    "max_current_sub": [feat_max_value, "current_sub"],
    "rms_current_main": [feat_rms_value, "current_main"],
    "rms_current_sub": [feat_rms_value, "current_sub"],
    "std_current_main": [feat_std_value, "current_main"],
    "std_current_sub": [feat_std_value, "current_sub"],
}

# location of the high_level csv that has been labelled with faile/not-failed labels
high_level_label_location = Path('high_level_LABELLED.csv')

# location of all the split signals (these are all the pickles that were created in the create_split_data.py)
split_data_folder = Path('interim_sample_data')

# read the high_level csv
df1 = pd.read_csv(high_level_label_location)
df1 = df1.dropna(subset=["failed"])  # drop rows that do not have a failed indicator

# Create the low-level df
# we will be calculating singlar values for the signals as well
df2 = low_level_df(
    split_data_folder,
    features,
    svd_feature=True,
    list_of_svd_signals=["current_sub", "current_main"],
    svd_feat_count=10,
    svd_window_size=10,
)

# label the individual cuts in the low-level df as failed or not
df_low = check_date(df1, df2)

# save as a csv
df_low.to_csv(("low_level_LABELLED.csv"), index=False)
print(df_low.head())
