"""
Script that is used to split raw cut signals into individual cuts. 
Cuts are split by tool number, and then by cut signal (whether the
cut signal is 0 or 1).

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
    low_level_df,
    split_df,
    check_date,
    extract_data_csv,
)

from feature_engineering import (
    feat_min_value,
    feat_max_value,
    feat_rms_value,
    feat_std_value,
)


# The folder where all the raw data is stored
raw_data_folder = Path("raw_sample_data")

# The folders within the raw_data_folder
# There will be multiple sub folders within the raw_data_folder
p1 = raw_data_folder / "NOV2019"
p2 = raw_data_folder / "SEPT2019"
p3 = raw_data_folder / "OCT2018"


path_list = [
    p1,
    p2,
    p3,
]

# interim data folder where the pickels of each split will be stored
# once they are created
interim_data_folder = Path("interim_sample_data")

# walk through each .csv, .mat, .pickle file, and separate the cuts
for p in path_list:

    l = [x[0] for x in os.walk(p)][1:]

    for cut_dir in l:

        file_names, new_load, cut_times = load_cut_files(cut_dir)

        # split up each new_load
        for cut_time in cut_times:

            some_cuts = split_df(new_load[cut_time], cut_time, stable_speed_only=True)
            single_tool_cuts = list(some_cuts.keys())
            for i in single_tool_cuts:
                name = "{}.pickle".format(str(i))

                pickle_out = open(interim_data_folder / name, "wb")
                pickle.dump(some_cuts[i], pickle_out)
                pickle_out.close()
