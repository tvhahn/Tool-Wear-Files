import numpy as np
import pandas as pd
import os
import re
import sys
from zipfile import ZipFile
import zipfile
import zlib
from pathlib import Path


# location of the zip folders containing the zip files of all the results
zip_path = Path("/home/tim/Documents/Tool-Wear-Files/temp")

# setup the location where the split cut data will be stored.
# folder location will be created if does not already exist
# Path("/home/tvhahn/scratch/interim_data_results").mkdir(parents=True, exist_ok=True)
# scratch_path = Path("/home/tvhahn/scratch/interim_data_results")
Path("interim_data_results").mkdir(parents=True, exist_ok=True)
scratch_path = Path("interim_data_results")


file_name = sys.argv[1]
file_folder_index = file_name.split(sep=".")[0]

# extract zip file
with ZipFile(zip_path / file_name, "r") as zip_file:
    # setup the location where the split cut data will be stored.
    # folder location will be created if does not already exist
    zip_file.extractall(path=(scratch_path / file_folder_index))

# location of all the split signals (these are all the pickles that were created in the create_split_data.py)
csv_result_folder = scratch_path / file_folder_index

# column prefixes
col_prefix = [
    "SGDClassifier",
    "KNeighborsClassifier",
    "LogisticRegression",
    "SVC",
    "RidgeClassifier",
    "RandomForestClassifier",
    "XGB",
    "LogisticRegression",
]

primary_cols_sorted = [
    "clf_name",
    "tool_list",
    "feat_list",
    "indices_to_keep",
    "info_no_samples",
    "info_no_failures",
    "info_no_feat",
    "to_average",
    "uo_method",
    "imbalance_ratio",
    "scaler_method",
    "parameter_sampler_seed",
    "initial_script_seed",
]

display_table_columns = [
    "clf_name",
    "tool_list",
    "feat_list",
    "indices_to_keep",
    "to_average",
    "auc_max",
    "auc_min",
    "auc_score",
    "auc_std",
    "f1_max",
    "f1_min",
    "f1_score",
    "f1_std",
    "precision",
    "precision_max",
    "precision_min",
    "precision_std",
    "recall",
    "recall_max",
    "recall_min",
    "recall_std",
    "roc_auc_max",
    "roc_auc_min",
    "roc_auc_score",
    "roc_auc_std",
]

for i, file in enumerate(os.listdir(csv_result_folder)):
    # open up the .csv file
    if file.endswith(".csv") and i == 0:
        df = pd.read_csv(csv_result_folder / file)
    elif file.endswith(".csv"):
        df = df.append(pd.read_csv(csv_result_folder / file), ignore_index=True, sort=False)
    
    # if i % 100 == 0:
    #     print('file no. ', i)
            
# print('Final df shape:', df.shape)

primary_cols = []
secondary_cols = []
for col in list(df.columns):
    if col.split('_')[0] in col_prefix:
        secondary_cols.append(col)
    elif col in primary_cols_sorted:
        pass
    else:
        primary_cols.append(col)

complete_col_list = primary_cols_sorted + sorted(primary_cols) + sorted(secondary_cols)
complete_col_list

primary_cols_sorted = primary_cols_sorted + sorted(primary_cols)

df = df[complete_col_list]



# create a save folder for the CSVs
Path("temp_csv_results").mkdir(parents=True, exist_ok=True)

df = df[(df['auc_min']>0.3)]

name_of_csv = "temp_csv_results/combined_results_{}.csv".format(str(file_folder_index))

# save as a csv
df.to_csv(name_of_csv,index=False)
