"""
Script that is used to create a labelled high_level csv.
The csv includes information on each cut (where each cut is the creation of one
brand new part). The csv can then be labelled based off of tool failure records.

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

# The folder where all the raw data is stored
raw_data_folder = Path("/home/tim/Documents/Checkfluid-Project/data/raw")
# raw_data_folder = Path('/home/tim/Documents/Checkfluid-Project/notebooks/1.7-tvh-refactor-pipeline/csv-mat_raw_data_sample')


# The folders within the raw_data_folder
# There will be multiple sub folders within the raw_data_folder
p1 = raw_data_folder / "NOV2019"
p2 = raw_data_folder / "SEPT2019"
p3 = raw_data_folder / "OCT2018_KS-NPC-6FF"
p4 = raw_data_folder / "NOV2018_KS-NPC-14N"
p5 = raw_data_folder / "JAN2019"
p6 = raw_data_folder / "FEB2019_KS-NPC-14N"

path_list = [
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
]

# create high-level CSV with all the cut names
high_level_csv(path_list, "high_level_master_20200107.csv")