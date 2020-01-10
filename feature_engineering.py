"""Create Features from Cut Signals

This module contains functions can be applied to cut signals to create 
features. Generally, these features will be appended onto a table that
contains information of all the cuts (e.g. low_level_cut_labels.csv).

The module structure is the following:


Feature Engineering Functions
=============================
feat_min_value : Calculates the minimum value of a signal

feat_max_value : Calculates the maximum value of a signal

feat_rms_value : Calculates the root-mean-square value of a signal

feat_std_value : Calculate the standard deviation of a signal


Other Feature Engineering Functions
===================================

- The "trajectory_matrix" function creates a trajectory matrix
that is used in signular value decomposition.

- The "svd_columns" function creates a dictionary of all the SVD
column names for each of the signals it is to be applied to.

- The "feat_svd_values" function calculates the singular values for
a signal.

"""

# Authors: Tim von Hahn <18tcvh@queensu.ca>
#
# License: MIT License


import numpy as np
import pandas as pd


def feat_min_value(df, signal_name):
    """Calculates the minimum value of a signal"""

    return np.min(df[signal_name].astype("float64"))


def feat_max_value(df, signal_name):
    """Calculates the maximum value of a signal"""

    return np.max(df[signal_name].astype("float64"))


def feat_rms_value(df, signal_name):
    """Calculates the root-mean-square value of a signal"""

    x = df[signal_name].astype("float64")
    return np.sqrt(np.mean(x ** 2))


def feat_std_value(df, signal_name):
    """Calculate the standard deviation of a signal"""

    return np.std(df[signal_name].astype("float64"))


# _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
"""
Functions for Singular Value Decomposition (SVD) calculations
"""
# _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#


def trajectory_matrix(s, window_size):
    """Create a trajectory matrix, that will be used forcalculating the SVD. 
    
    The trajectory matrix is also implemented in paper by Liu et al. 2014,
    http://bit.ly/2tvXp9A   

    Parameters
    ===========
    s : numpy array
        The numpy array of the signal

    window_size : int
        The size of the window that you want the trajectory matrix to go
        over.

    Returns
    ===========
    a : numpy array
        Returns the numpy array trajectory matrix

    """

    # resize time series so it is divisible by window size
    s = s[0 : (s.shape[0] - s.shape[0] % window_size)]

    k = len(s) - window_size

    a = np.zeros((k, window_size))

    for i in range(k):
        a[i] = s[i : i + window_size]

    return a


def svd_columns(feature_count, list_of_signals=["current_sub", "current_main"]):
    """Creates a dictionary of all the SVD column names for each of the signals
    it is to be applied to.

    Parameters
    ===========
    feature_count : int
        The number of singular-value-decomposition values to be included
        for each signal

    list_of_signals : list
        List that contains all the signal names for which the SVD values should
        be calculated for

    Returns
    ===========
    svd_feature_dictionary : dict
        Returns a dictionary with the column names as keys.
    
    """

    # create empty dictionary
    svd_feature_dictionary = {}

    # go through each of the signals in the list_of_signals
    # and add the appropriate column name to the dictionary
    for i in list_of_signals:
        for j in range(feature_count):
            svd_feature_dictionary["svd_" + i + "_{}".format(j)] = [i, j]

    return svd_feature_dictionary


def feat_svd_values(svd_feature_dictionary, df, window_size):
    """Calculate singular values of a signal.

    Parameters
    ===========
    svd_feature_dictionary : dict
        A dictionary that has all the SVD column names in it, along
        with the signals that the SVD will be applied to.

    df : pandas dataframe
        Dataframe that includes the all the signals that the SVD will be
        applied on

    window_size : int
        The size of the window that you want the trajectory matrix to go
        over.

    Returns
    ===========
    svd_feature_dictionary : dict
        Returns the dictionary with the SVD values included.
    
    """

    # get unique values that are in the svd feature dictionary
    signals_svd = []
    for k in svd_feature_dictionary:
        signals_svd.append(svd_feature_dictionary[k][0])

    signals_svd = set(signals_svd)

    feature_count = int(len(svd_feature_dictionary) / len(signals_svd))

    # build trajectory matrix
    for k in signals_svd:

        # try and create a trajectory matrix
        # if it does not work, say because the cut size is less than
        # the window size, then it will simply output "blanks"
        try:
            s = df[k].to_numpy("float32")

            # build trajectory matrix
            a = trajectory_matrix(s, window_size)

            # use numpy linalg.svd to find the singular values
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
            singular_vals = np.linalg.svd(a, full_matrices=False)

            # only include the first x number of singular values
            # (the same number that matches up with the number of
            # features in the svd_feature_dictionary)
            singular_vals = singular_vals[1][:feature_count]

            for j in range(feature_count):
                s_key = "svd_" + k + "_{}".format(j)
                svd_feature_dictionary[s_key].append(singular_vals[j])

        except:
            # print("#*#*ERROR#*#*#")  # print off an error

            # make sure we put "blanks" in for each of the
            # features if the SVD calculation does not work
            singular_vals = [""] * feature_count

            for j in range(feature_count):
                s_key = "svd_" + k + "_{}".format(j)
                svd_feature_dictionary[s_key].append(singular_vals[j])

    return svd_feature_dictionary
