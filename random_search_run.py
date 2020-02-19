import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import random

from sklearn.model_selection import ParameterSampler
from scipy.stats import randint as sp_randint
from scipy.stats import uniform

from functions import (
    under_over_sampler,
    classifier_train,
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
sampler_seed = random.randint(0,2**16)
no_iterations = 200

# create list of tools that we want to look over
# these are only the tools that we know we have wear-failures
tool_list_all = [57, 54, 32, 36, 22, 8, 2]

# other parameters
scaler_methods = ["standard", "min_max"]
imbalance_ratios = [0.1, 0.3, 0.5, 0.7, 0.8, 1]
average_across_indices = [True, False]


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
    None,
]



#############################################################################
# start by loading the csv with the features
file_folder = Path(
    "/home/tim/Documents/Checkfluid-Project/data/processed/"
    "_tables/low_level_labels_created_2020-01-30"
)

file = file_folder / "low_level_labels_created_2020-01-27.csv"

df = pd.read_csv(file)

# sort the values by date and index so that it is reproducible
df = df.sort_values(by=["unix_date", "tool", "index"])

# replace NaN's in failed columns with 0
df["failed"].fillna(
    0, inplace=True, downcast="int"
)  # replace NaN in 'failed' col with 0

# create a test and train set
df_test = df[df["date"] >= "2019-02-08"].copy().reset_index(drop=True)
df_train = (
    df[df["date"] < "2019-02-22"].copy().reset_index(drop=True)
)  # This is our train set

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
    "no_tools": sp_randint(1, len(tool_list_all)),
    "no_feat": sp_randint(1, len(feat_generic_all)),
    "no_indices": sp_randint(1, 19),
    "classifier_used": classifier_list_all,
    "average_across_index": average_across_indices,
    "uo_method": over_under_sampling_methods,
    "scaler_method": scaler_methods,
    "parameter_sampler_random_int": sp_randint(0, 2 ** 16),
    "imbalance_ratio": imbalance_ratios,
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

for i, p in enumerate(p_list):

    # set random.seed
    random.seed(p["parameter_sampler_random_int"])

    # get specific parameters
    clf_name = str(p["classifier_used"]).split(" ")[1]
    tool_list = sorted(random.sample(tool_list_all, p["no_tools"]))
    feat_list = sorted(random.sample(feat_generic_all, p["no_tools"]))
    indices_to_keep = sorted(random.sample(range(0, 19), p["no_indices"]))
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
    X_train, y_train, df1 = get_xy_from_df(
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
        indices_to_keep = sorted(random.sample(range(0, 19), p["no_indices"]))
        tool_list = sorted(random.sample(tool_list_all, p["no_tools"]))
        # print("Revised Indices: ", indices_to_keep)
        # print("Revised tool_list: ", tool_list)

        X_train, y_train, df1 = get_xy_from_df(
            df_train,
            tool_list=tool_list,
            indices_to_keep=indices_to_keep,
            to_average=to_avg,
            generic_feat_list=feat_list,
        )

        parameter_values['indices_to_keep'] = indices_to_keep
        parameter_values['tool_list'] = tool_list

        len_data = len(y_train)
        no_label_failed = np.sum(y_train)
        seed_indexer += 1

    parameter_values['info_no_samples'] = len_data
    parameter_values['info_no_failures'] = no_label_failed

    # save the general parameters values
    df_gpam = pd.DataFrame.from_dict(parameter_values, orient="index").T

    # instantiate the model
    clf, classifier_parameters = clf_function(parameter_sampler_random_int)

    # save classifier parameters into dataframe
    df_cpam = pd.DataFrame.from_dict(classifier_parameters, orient="index").T

    # train the model
    try:
        result_dict, _, _ = classifier_train(
            X_train,
            y_train,
            clf,
            scaler_method=scaler_method,
            uo_sample_method=uo_method,
            imbalance_ratio=imbalance_ratio,
            train_on_all=False,
            print_results=True,
        )


        df_result_dict = pd.DataFrame.from_dict(result_dict, orient="index").T
        df_result_dict.astype('float16').dtypes

        if i == 0:
            df_results = pd.concat([df_gpam, df_cpam, df_result_dict], axis=1)
        else:
            df_results = df_results.append(
                pd.concat([df_gpam, df_cpam, df_result_dict], axis=1)
            )

        if i % 50 == 0:
            df_results.to_csv('temp_result_{}.csv'.format(str(date_time)),index=False)

    except ValueError as err:
        print(err)
        print('#!#!#!#!#! SKIPPING')
        pass
    except:
        pass

df_results.to_csv('temp_result_{}.csv'.format(str(date_time)),index=False)
