import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import random
import sys

from sklearn.model_selection import ParameterSampler
from scipy.stats import randint as sp_randint
from scipy.stats import uniform

from functions import (
    under_over_sampler,
    classifier_train,
    classifier_train_manual,
    make_generic_df,
    get_xy_from_df,
    plot_precision_recall_vs_threshold,
    plot_precision_vs_recall,
)

from classification_methods import (
    random_forest_classifier,
    knn_classifier,
    logistic_regression,
    sgd_classifier,
    ridge_classifier,
    svm_classifier,
    gaussian_nb_classifier,
    xgboost_classifier,
)

# stop warnings from sklearn
# https://stackoverflow.com/questions/32612180/eliminating-warnings-from-scikit-learn
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

# to profile script for memory usage, use:
# /usr/bin/time -f "mem=%K RSS=%M elapsed=%E cpu.sys=%S .user=%U" python random_search_run.py
# from https://unix.stackexchange.com/questions/375889/unix-command-to-tell-how-much-ram-was-used-during-program-runtime

#############################################################################
# RANDOM SEARCH PARAMETERS
# fill these out to set parameters for the random search

# set a seed for the parameter sampler
sampler_seed = random.randint(0, 2 ** 16)
no_iterations = 30000

# create list of tools that we want to look over
# these are only the tools that we know we have wear-failures [57, 54, 32, 36, 22, 8, 2]
tool_list_all = [57, 54, 32, 36, 22, 8, 2]
tool_list_some = [57, 32, 22, 8, 2, 36]

# other parameters
scaler_methods = ["standard", "min_max"]
imbalance_ratios = [0.5,0.8,1]
average_across_indices = [True,False]


# list of classifiers to test
classifier_list_all = [
    random_forest_classifier,
    knn_classifier,
    logistic_regression,
    sgd_classifier,
    ridge_classifier,
    svm_classifier,
    gaussian_nb_classifier,
    xgboost_classifier,
]

over_under_sampling_methods = [
    "random_over",
    "random_under",
    "random_under_bootstrap",
    "smote",
    "adasyn",
    # None,
]

# no cut indices past 9 that are valid
index_list = [
    list(range(0, 10)),
    list(range(1, 10)),
    list(range(1, 9)),
    list(range(1, 8)),
    list(range(2, 8)),
    list(range(3, 7)),
    list(range(2, 9)),
    list(range(2, 10)),
]

#############################################################################
# test and train folds
# failures for tool 54 on following dates:
    # 2018-11-15
    # 2019-01-28
    # 2019-01-29
    # 2019-01-30
    # 2019-02-04
    # 2019-02-07
    # 2019-02-08
    # 2019-09-11 - These are resampled into pickle files (in case that matters)
    # 2019-11-27
    # 2019-01-23 - These are from January data without speed

test_fold = [
    "2018-10-23",
    "2018-11-15", # failures
    "2018-11-16",
    "2018-11-19",
    "2019-09-11", # failures
    "2019-09-13",
]

train_fold_1 = [
    "2018-11-21", 
    "2019-01-25", 
    "2019-01-28", # failures
    "2019-11-27", # failures
    "2019-01-23", # failures, from Jan without speed
    "2019-05-03",
    ]

train_fold_2 = [
    "2019-01-29", # failures
    "2019-01-30", # failures
    "2019-02-01",
    "2019-02-08", # failures
    "2019-09-10",
    "2019-09-12",
    "2018-11-20",
    "2019-02-11",
    "2019-01-22", # from Jan withough speed
    "2019-05-04",
]

train_fold_3 = [
    "2019-02-04", # failures
    "2019-02-05", 
    "2019-02-07", # failures
    "2019-05-06",
    "2019-01-22", # from Jan without speed  
    ]

train_folds = [train_fold_1, train_fold_2, train_fold_3]
train_dates_all = [date for sublist in train_folds for date in sublist]


#############################################################################
# start by loading the csv with the features
# file_folder = Path(
#     "/home/tim/Documents/Checkfluid-Project/data/processed/"
#     "_tables/low_levels_labels_created_2020-03-11"
# )

# for HPC
file_folder = Path(
    "/home/tvhahn/projects/def-mechefsk/tvhahn/_tables/low_level_labels_created_2020-03-11"
)

file = file_folder / "low_level_labels_created_2020.03.11_v3.csv"

df = pd.read_csv(file)

# sort the values by date and index so that it is reproducible
df = df.sort_values(by=["unix_date", "tool", "index"])

# replace NaN's in failed columns with 0
df["failed"].fillna(
    0, inplace=True, downcast="int"
)  # replace NaN in 'failed' col with 0

# function to convert pandas column to datetime format
def convert_to_datetime(cols):
    unix_date = cols[0]
    value = datetime.fromtimestamp(unix_date)
    return value


# apply 'date_ymd' column to dataframe
df["date"] = df[["unix_date"]].apply(convert_to_datetime, axis=1)
# convert to a period, and then string
df["date_ymd"] = pd.to_datetime(df["date"], unit="s").dt.to_period("D").astype(str)


# create train set
df_train = df[df["date_ymd"].isin(train_dates_all)].reset_index(drop=True).copy()

#############################################################################
# build the parameters to search over
# start with building the generic feature list which we will sample from
feat_generic_all = []

for feat in list(df_train.columns):
    if "sub" in feat:
        feat_generic_all.append(feat.replace("_sub", ""))
    else:
        pass

# parameter dictionary for random sampler to go over
parameters_sample_dict = {
    "no_tools": sp_randint(0, len(tool_list_some)),
    "no_feat": sp_randint(1, 25), # sp_randint(1, len(feat_generic_all))
    "classifier_used": classifier_list_all,
    "average_across_index": average_across_indices,
    "uo_method": over_under_sampling_methods,
    "scaler_method": scaler_methods,
    "parameter_sampler_random_int": sp_randint(0, 2 ** 16),
    "imbalance_ratio": imbalance_ratios,
    # additional parameters to narrow down random search
    "index_list": index_list,
}

# generate the list of parameters to sample over
p_list = list(
    ParameterSampler(
        parameters_sample_dict, n_iter=no_iterations, random_state=sampler_seed
    )
)

#############################################################################
# run models with each of the parameters

date_time = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")

for k, p in enumerate(p_list):

    # set random.seed
    random.seed(p["parameter_sampler_random_int"])

    # get specific parameters
    clf_name = str(p["classifier_used"]).split(" ")[1]

    tool_list = sorted(
        random.sample(tool_list_some, p["no_tools"])
        + [54])

    # tool_list = sorted(
    #     [54]
    #     + random.sample([36], random.randint(0, 1))
    # )

    feat_list = sorted(random.sample(feat_generic_all, p["no_feat"]))
    indices_to_keep = p["index_list"]
    to_avg = p["average_across_index"]
    uo_method = p["uo_method"]

    # if svm, need to prevent too large a dataset, thus will only use undersampling
    if clf_name == "svm_classifier":
        uo_method = random.sample(["random_under", "random_under_bootstrap"], 1)

    imbalance_ratio = p["imbalance_ratio"]
    scaler_method = p["scaler_method"]
    parameter_sampler_random_int = p["parameter_sampler_random_int"]
    clf_function = p["classifier_used"]

    # build dictionary to store parameter results and other info
    parameter_values = {
        "clf_name": clf_name,
        "tool_list": tool_list,
        "feat_list": feat_list,
        "indices_to_keep": indices_to_keep,
        "info_no_samples": None,
        "info_no_failures": None,
        "info_no_feat": p["no_feat"],
        "to_average": to_avg,
        "uo_method": uo_method,
        "imbalance_ratio": imbalance_ratio,
        "scaler_method": scaler_method,
        "parameter_sampler_seed": parameter_sampler_random_int,
        "initial_script_seed": sampler_seed,
    }

    # print('original indices_to_keep:', indices_to_keep)
    # print('original tool_list:', tool_list)

    # prepare the data table
    X_train, y_train, df_ymd_only = get_xy_from_df(
        df_train,
        tool_list=tool_list,
        indices_to_keep=indices_to_keep,
        to_average=to_avg,
        generic_feat_list=feat_list,
    )

    # check if empty X_train
    len_data = len(y_train)
    # check if not enough labels in y_train
    no_label_failed = np.sum(y_train)

    seed_indexer = 0
    while len_data < 20 or no_label_failed < 15:
        random.seed(p["parameter_sampler_random_int"] + seed_indexer)
        tool_list = sorted(
            random.sample(tool_list_some, p["no_tools"])
            + random.sample([54, 36], random.randint(1, 2))
        )
        # print("Revised Indices: ", indices_to_keep)
        # print("Revised tool_list: ", tool_list)

        X_train, y_train, df_ymd_only = get_xy_from_df(
            df_train,
            tool_list=tool_list,
            indices_to_keep=indices_to_keep,
            to_average=to_avg,
            generic_feat_list=feat_list,
        )

        parameter_values["tool_list"] = tool_list

        len_data = len(y_train)
        no_label_failed = np.sum(y_train)
        seed_indexer += 1

    parameter_values["info_no_samples"] = len_data
    parameter_values["info_no_failures"] = no_label_failed

    # save the general parameters values
    df_gpam = pd.DataFrame.from_dict(parameter_values, orient="index").T

    # instantiate the model
    clf, classifier_parameters = clf_function(parameter_sampler_random_int)

    # save classifier parameters into dataframe
    df_cpam = pd.DataFrame.from_dict(classifier_parameters, orient="index").T

    # train the model
    try:
        result_dict, _, _ = classifier_train_manual(
            X_train,
            y_train,
            df_ymd_only,
            train_folds,
            clf,
            scaler_method=scaler_method,
            uo_sample_method=uo_method,
            imbalance_ratio=imbalance_ratio,
            train_on_all=False,
            print_results=True,
        )

        df_result_dict = pd.DataFrame.from_dict(result_dict, orient="index").T
        df_result_dict.astype("float16").dtypes

        if k == 0:
            df_results = pd.concat([df_gpam, df_cpam, df_result_dict], axis=1)
        else:
            df_results = df_results.append(
                pd.concat([df_gpam, df_cpam, df_result_dict], axis=1)
            )

        # save directory for when on the HPC
        save_directory = Path('/home/tvhahn/scratch/_temp_random_search_results')
        # save_directory = Path("temp_results/")

        file_save_name = "temp_result_{}_{}_{}.csv".format(
            str(date_time), str(sys.argv[1]), str(sampler_seed)
        )
        if k % 10 == 0:
            df_results.to_csv(save_directory / file_save_name, index=False)

    except ValueError as err:
        print(err)
        print("#!#!#!#!#! SKIPPING")
        pass
    except:
        pass

df_results.to_csv(save_directory / file_save_name, index=False)

