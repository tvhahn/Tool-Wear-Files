"""Load Raw Cut Files

VERSION 1.1.0

### CHANGES IN VERSION 1.1.0

2020.03.06:
    - Added new feature engineering functions (feat_freq_pk_s2, feat_crest_factor,
    feat_tdh_estimate) to feature_engineering.py

###

This module contains functions that take raw cut files from a Fanuc CNC machine
and output a standard format as a pandas dataframe. Additional helper functions
are also included.

The module structure is the following:

- The "extract_data_csv" function takes an individual cut file, in the form
of a csv, extracts the appropriate columns, and outputs it in a standard 
pandas dataframe.

- The "extract_data_pickle" function takes an individual cut file, in the form
of a pickle, and outputs it in a standard pandas dataframe. Often, the pickle
files of cuts have been created after resampling a cut file taken at 2000Hz to
1000Hz.

- The "extract_data_mat" function takes an individual cut file, in the form
of a matlab .mat file, extracts the appropriate columns, and outputs it in a 
standard pandas dataframe.

- The "load_cut_files" function takes all the raw cut files (either csv, mat,
or pickles) and returns a dictionary of all the cuts -- with each cut being 
represented as a pandas dataframe. This function uses the "extract_data_csv",
 "extract_data_mat", or "extract_data_mat" functions.

- The "high_level_csv" function walks through multiple directories (and their 
sub-directories) containing raw cut files. The function create a "high-level" 
csv file that includes the information for each cut (such as the time and 
date). The csv can then be hand-labeled to identify which cuts have failures 
associated with them. This information is then used by the "low_level_df" 
function to create a detailed dataframe the includes labelled data for all the
cuts performed by a individual tool.

- The "stable_speed_section" function takes a cut dataframe and returns the
portion of the cut that is in a stable speed region.

- The "split_df" function takes a dataframe of a cut and splits it up by tool 
number and when the cut-signal is on. It returns a dictionary of all the 
split cuts with their unique names.

- The "low_level_df" function walks through each of the processed cut files 
(these are pickle cut files split by tool number, and cut-signal) and 
calculates features for each cut (features such as RMS, Kurtosis, SVD, etc.). 
It then returns all the features, along with the unique cut names, as a 
dataframe. Using the high-level csv the cuts in the dataframe can be labeled.

- The "check_date" function takes the low_level dataframe, and the high_level
dataframe, and assigns a label (as either failed or not) for each cut in the 
low_level dataframe.

- The "rename_cols_df" function applies a standard naming convention to a
dataframe of the cuts.

- The "cut_signal_apply" function takes the PMC signal (when using Servoview
to collect the data), and determines if the tool is in cut or not. This is a 
helper function that is used in the "extract_data_csv" function.

- The "tool_no_apply" function takes the PMC signal and determines which tool
is in cut. This is a helper function that is used in the "extract_data_csv"
function.

"""

# Authors: Tim von Hahn <18tcvh@queensu.ca>
#
# License: MIT License


import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path
import os
import time
from datetime import datetime
import logging
import pickle
from feature_engineering import svd_columns, feat_svd_values, calc_fft
from scipy import stats


def extract_data_csv(file):
    """Extracts the useful columns from a csv file. The csv is a standard output from
    the Servoviewer application.

    Output is a pandas dataframe and the unix timestamp.

    Parameters
    ===========
    file : string
        The csv name that contains the cut info. e.g. auto$002.csv

    Returns
    ===========
    df : dataframe
        Pandas dataframe. Columns such as: current_main, current_sub, power_main,
        power_sub, cut_signal, speed_sub, tool_no, current_main, etc.

    unixtime : int
        Unix timestamp of when the cut started. 
        See wikipedia for detail: https://en.wikipedia.org/wiki/Unix_time
        Example: 2018-10-23 08:45:55 --> 1540298755
    
    Future Work / Improvements
    ==========================
    - Rather than using pandas 'apply' function (which operates row-by-row), 
    perform a vector operation (which operates across the entire vector at once,
    thus is much faster).
    
    """

    # get the unix timestamp for when the file was modified (http://bit.ly/2RW5cYo)
    unixtime = int(os.path.getmtime(file))

    # dictionary that contains the name of columns and their native format
    dict_int16_columns = {
        "speed_main": "int16",
        "speed_sub": "int16",
        "current_main": "int16",
        "current_sub": "int16",
        "cut_signal": "int16",
        "tool_no": "int16",
    }

    # load the csv
    # don't load certain rows as they contain meta info
    df = pd.read_csv((file), skiprows=[0, 1, 3], dtype="float32")

    df["cut_signal"] = df[["PMC"]].apply(cut_signal_apply, axis=1)  # get cut signal
    df["tool_no"] = df[["PMC"]].apply(tool_no_apply, axis=1)  # get tool_no
    df.drop(["Index", "Time", "PMC"], axis=1, inplace=True)  # remove the "index" column

    # standardize column names
    df = rename_cols_df(df)

    # cast some of the columns in their native datatype
    df = df.astype(dict_int16_columns)

    return (df, unixtime)


def extract_data_pickle(file):
    """Extracts the useful columns from a pickle file.

    Output is a pandas dataframe and the unix timestamp.

    The pickle file is often used when downsampling from 2000hz to 1000hz.

    Parameters
    ===========
    file : string
        The pickle file name containing the cut info. e.g. 1568213682.pickle

    Returns
    ===========
    df : dataframe
        Pandas dataframe. Columns such as: current_main, current_sub, power_main,
        power_sub, cut_signal, speed_sub, tool_no, current_main, etc.

    unixtime : int
        Unix timestamp of when the cut started. 
        See wikipedia for detail: https://en.wikipedia.org/wiki/Unix_time
        Example: 2018-10-23 08:45:55 --> 1540298755
    
    Future Work / Improvements
    ==========================
    - Rather than using pandas 'apply' function (which operates row-by-row), 
    perform a vector operation (which operates across the entire vector at once,
    thus is much faster).
    
    """

    # dictionary that contains the name of columns and their native format
    dict_int16_columns = {
        "speed_main": "int16",
        "speed_sub": "int16",
        "cut_signal": "int16",
        "tool_no": "int16",
    }

    # load the pickle file
    with open(file, "rb") as input_file:
        d = pickle.load(input_file)

    unixtime = list(d.keys())[0]  # get the unix timestamp of the cut
    df = d[unixtime][0]
    df = df.astype(dtype="float32")  # cast columns as float32
    df = rename_cols_df(df)  # standardize column names
    try:
        df.drop(["time"], axis=1, inplace=True)  # remove the "index" column
    except:
        pass

    # cast some of the columns in their native datatype
    try:
        df = df.astype(dict_int16_columns)
    except:
        dict_int16_columns = {
            "cut_signal": "int16",
            "tool_no": "int16",
        }

        df = df.astype(dict_int16_columns)

    return (df, unixtime)


def extract_data_mat(m):
    """Extracts the useful columns from a matlab file.

    Output is a pandas dataframe and the unix timestamp.

    Parameters
    ===========
    m : dict
        Takes a dictionary created by the scipy.io.loadmat function
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html

    Returns
    ===========
    df : dataframe
        Pandas dataframe. Columns: current_main, current_sub, power_main,
        power_sub, cut_signal, speed_sub, tool_no, current_main, 
        voltage_main, voltage_sub

    unixtime : int
        Unix timestamp of when the cut started. 
        See wikipedia for detail: https://en.wikipedia.org/wiki/Unix_time
        Example: 2018-10-23 08:45:55 --> 1540298755

    """

    # Record the timestamp. Use try as two different formats
    try:
        time_stamp = m["TimeStamp"]
    except:
        pass

    try:
        time_stamp = m["Date_Time"]
    except:
        pass

    # Convert the timestamp to unix-time and date-time formats
    for x in time_stamp:
        t = np.array2string(x)

        # Convert string to datetime http://bit.ly/2WGCZnL
        d = datetime.strptime(t, "'%d-%b-%Y %H:%M:%S'")

        # Convert to unix time http://bit.ly/2GJMf50
        unixtime = int(time.mktime(d.timetuple()))

    # Tuple of columns to remove
    entries_remove = (
        "__header__",
        "__version__",
        "__globals__",
        "TimeStamp",
        "Date_Time",
        "Index",
        "Time",
        "Tool_Offset_Data",
        "Pulsecode_POsition_X",
        "Pulsecode_POsition_Z",
        "PhaseA_SubSpindle",
        "PhaseA_MainSpindle",
    )

    # Dict of columns names to standard names
    col_name_change = {
        "Current_MainSpindle": "current_main",
        "Current_SubSpindle": "current_sub",
        "Power_MainSpindle": "power_main",
        "Power_SubSpindle": "power_sub",
        "CUT_Signal": "cut_signal",
        "Cut_Signal": "cut_signal",
        "Speed_SubSpindle": "speed_sub",
        "Speed_MainSpindle": "speed_main",
        "TOOL_Number": "tool_no",
        "Tool_Number": "tool_no",
        "INORM_MainSpindle": "current_main",
        "INORM_SubSpindle": "current_sub",
        "LMDAT_MainSpindle": "voltage_main",
        "LMDAT_SubSpindle": "voltage_sub",
        "TorqueCommand_Z": "tcmd_z",
        "Psition_ERR_Z": "error_z",
        "Psition_ERR_X": "error_x",
    }

    # Remove the unnecessary columns from the data-set
    # (stackoverflow link: https://bit.ly/2D8wcNU)
    for k in entries_remove:
        m.pop(k, None)

    # Create a list of the column names
    index_list = []
    for k in m:
        index_list.append(k)

    # Create a dummy pandas df to instantiate the dataframe
    # could probably fix this up in later versions...
    df = pd.DataFrame(np.reshape(m[index_list[1]], -1), columns=["dummy"])

    # Iterate through the variables and append onto our 'dummy' df
    for k in m:
        df[k] = pd.DataFrame(np.reshape(m[k], -1))
    df = df.drop(["dummy"], axis=1)  # Drop the 'dummy' column

    # Rename columns to standard names
    df.rename(columns=col_name_change, inplace=True)

    # dictionary that contains the native data types of certain columns
    dict_int16_columns = {
        "speed_main": "int16",
        "speed_sub": "int16",
        "current_main": "int16",
        "current_sub": "int16",
        "cut_signal": "int16",
        "tool_no": "int16",
    }

    # cast all columns as float32 (as opposed to float64)
    df = df.astype("float32")

    # cast some of the columns in their native datatype
    try:
        df = df.astype(dict_int16_columns)
    except:
        dict_int16_columns = {
            "current_main": "int16",
            "current_sub": "int16",
            # "cut_signal": "int16",
            "tool_no": "int16",
        }
        df = df.astype(dict_int16_columns)

    return (df, unixtime)


def load_cut_files(sub_directory):
    """Load all .mat, .csv, or .pickle files in a sub-directory and return the extracted data into a dictionary
    
    Parameters
    ===========
    sub_directory : string
        Location of the sub_directory with .mat files.
        Example: '/Checkfluid-Project/data/raw/OCT2018/23OCT2018_KS-NPC-6FF'

    Returns
    ===========
    file_names : dict
        Dictionary of all the matlab file names associated with their unix timestamp labels

    data : dict
        Dictionary containing the dataframe of each cut in the sub-directory

    cut_times : list
        Sorted list of all the unix timestamps
    
    Examples
    ===========
    >>> from load_data import load_cut_files
    >>> from pathlib import Path
    >>> raw_data_folder = Path('raw_data_sample/')
    >>> sub_folder = raw_data_folder / 'OCT2018_KS-NPC-6FF/23OCT2018_KS-NPC-6FF'
    >>> file_names, data, cut_times = load_cut_files(sub_folder)
    Load file:  Data_KS-NPC-6FF_0.mat
    Load file:  Data_KS-NPC-6FF_1.mat
    >>> print(file_names)
    {1540298755: 'Data_KS-NPC-6FF_0.mat', 1540298934: 'Data_KS-NPC-6FF_1.mat'}
    >>> print(data)
    {1540298755:         cut_signal  current_main  current_sub  power_main  \ ...
    0                0             3            1   -0.039062   0.001221   
    1                0             2            1    0.002441   0.002441   
    ... }
    >>> print(cut_times)
    [1540298755, 1540298934]

    """

    data = {}  # empty dict for the data
    cut_times = []  # empty list for the cut-times
    file_names = {}  # empty dict for the file names

    # iterate through the sub_directory and open up
    # applicable files (.mat, .csv, .pickle)
    for file in os.listdir(sub_directory):

        # open up the .mat file
        if file.endswith(".mat"):
            data_location = os.path.join(sub_directory, file)
            print("Load file: ", file)
            m = sio.loadmat(data_location)  # use scipy to open matlab file
            df, t = extract_data_mat(m)
            cut_times.append(t)
            file_names[t] = file

            # Append new dataframe into dictionary with a timestamp key
            # timestamp key is unixtime
            data[t] = df

        # open up .csv file
        elif file.endswith(".csv"):
            data_location = os.path.join(sub_directory, file)
            print("Load file: ", file)
            df, t = extract_data_csv(data_location)
            cut_times.append(t)
            file_names[t] = file

            # Append new dataframe into dictionary with a timestamp key
            # timestamp key is unixtime
            data[t] = df

        # open up .pickle file
        elif file.endswith(".pickle"):
            data_location = os.path.join(sub_directory, file)
            print("Load file: ", file)
            df, t = extract_data_pickle(data_location)
            cut_times.append(t)
            file_names[t] = file

            # Append new dataframe into dictionary with a timestamp key
            # timestamp key is unixtime
            data[t] = df

        else:
            pass

    cut_times = sorted(cut_times)  # Sort the cut-times from earliest to latest
    return (file_names, data, cut_times)


def high_level_csv(path_list, csv_name):
    """Create a high-level csv listing each cut in the data set

    Once created, each cut in the high_level csv can be labelled as to whether
    it has failed, or not (or use whatever label you want).
    
    Parameters
    ===========
    path_list : list
        List of all the main directories that contain .mat, .csv, or .pickle files. Each 
        main directory contains several sub directories.

    csv_name : string
        Name of the intended high_level csv. File name must end with '.csv' (e.g. 'my_high_level_file.csv')

    Returns
    ===========
    file_name.csv : csv
        Creates a csv in the directory where script is being run.
    
    Examples
    ===========
    >>> from load_data import high_level
    >>> from pathlib import Path
    >>> raw_data_folder = Path('raw_data_sample/')
    >>> p1 = raw_data_folder / 'OCT2018_KS-NPC-6FF'
    >>> p2 = raw_data_folder / 'JAN2019_KS-NPC-916U'
    >>> path_list = [p1,p2]
    >>> high_level_csv(path_list, 'high_level_test.csv')

    (creates a 'high_level_test.csv' in directory)
    
    """

    # open up a csv to write to
    f = open(csv_name, "w")

    # column names of the csv
    columns_high = [
        "unix_date",
        "date",
        "cut_dir",
        "part",
        "file_name",
        "tools",
        "len_cut",
        "no_points",
        "signals_names",
        "failed",
        "failed_tools",
        "comment",
    ]

    # write the columns names to the csv
    for i in columns_high:
        f.write(i + ",")
    f.write("\n")

    # Create list of sub-directories http://bit.ly/2GTn9Rl
    for p in path_list:

        # list of all the sub-directories
        l = [x[0] for x in os.walk(p)][1:]

        paths = [p]

        # Open up each sub-directory and load all the cut files. Then record
        # information about each cut in the csv.
        for cut_dir in l:

            relative_paths = [
                os.path.relpath(cut_dir, p) for p in paths
            ]  # Get relative path: http://bit.ly/2vrcC9E

            print("######## ", relative_paths[0], " ########")

            # load each cut in the sub-directory
            file_names, new_load, cut_times = load_cut_files(cut_dir)

            # go through each of the cuts in the sub-directory
            for t in cut_times:

                time_stamp = str(
                    datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
                )

                d = time_stamp.split(" ")[0]
                h = time_stamp.split(" ")[1]
                f.write(str(t) + ",")
                f.write(time_stamp + ",")
                f.write(str(cut_dir) + ",")

                try:
                    f.write(r[1] + ",")
                except:
                    f.write("" + ",")
                f.write(file_names[t] + ",")

                # create a string with the tools used in cut
                tool_list = [x for x in new_load[t].tool_no.unique()]
                tool_list.sort()
                for tool in tool_list:
                    f.write(str(tool) + " ")
                f.write(",")

                # n samples in cut, along with time of cut
                # time_cut = new_load[t]["time_sec"].iloc[-1]
                len_cut = len(new_load[t])
                time_cut = len_cut / 1000

                list_columns = list(new_load[t].columns)
                list_columns.sort()

                f.write(str(time_cut) + ",")
                f.write(str(len_cut) + ",")
                for i in list_columns:
                    f.write(i + " ")

                f.write("\n")

            print(new_load[cut_times[0]].columns.values)

    f.close()


def stable_speed_section(df):
    """Identifies the stable speed region in a cut. Returns a dataframe with 
    only the stable speed region.

    The function takes a standard cut dataframe. I then finds the most common
    speed value (by using the numpy mode function). It then windows off the 
    cut between the first and last instance of the mode.


    Parameters
    ===========
    df : pandas dataframe
        A standard cut dataframe that includes both sub and main spindle speeds


    Returns
    ===========
    df : pandas dataframe
        Dataframe with only the region that is identified as "stable speed"


    Future Work / Improvements
    ==========================
    -Rather than using "speed" to check which spindle is most active,
    should we use something like current instead?

    """

    def find_stable_speed(df, speed_array):

        # get absolute value of the speed array
        speed_array = np.abs(speed_array)

        # find the most common speed value (the mode)
        mode_speed = stats.mode(speed_array)[0][0]

        # find the index of the most commons speed value
        percentage_val = 0.005
        l = np.where(
            (speed_array > (mode_speed - mode_speed * percentage_val))
            & (speed_array < (mode_speed + mode_speed * percentage_val))
        )[0]

        # now create the dataframe that only includes the range
        # of indices where the most common speed values are
        df = df[l[0] : l[-1]]

        return df

    df = df.reset_index()

    # check to see if the sub or main spindle is the one most active
    if np.abs(df["speed_sub"].mean()) > np.abs(df["speed_main"].mean()):
        speed_array = df["speed_sub"].to_numpy()
        df = find_stable_speed(df, speed_array)

    else:
        speed_array = df["speed_main"].to_numpy()
        df = find_stable_speed(df, speed_array)

    return df


def split_df(df, cut_time, stable_speed_only=False):
    """Load a dataframe of a cut and split it up by tool # and cut_signal==true
    
    Parameters
    ===========
    df : pandas dataframe
        Dataframe of a single cut

    cut_time : int
        Unix timestamp of the cut. Will be converted to a string.

    Returns
    ===========
    dict_cuts : dictionary
        Dictionary of the split cuts cut -- labeled by timestamp, tool number, and sequence 
        e.g. {1548788710_22_0: ...data... , 1548788710_22_1: ...data... }

    """

    print("Starting split for:\t", str(cut_time))

    # use np.split to split dataframe based on condition: http://bit.ly/2GEFBwr
    split_data = np.split(df, *np.where(df.cut_signal == 0))

    # empty dictionary to store the individual tool cuts and index
    dict_cuts = {}
    tool_index = {}

    for x in split_data:

        if len(x) > 1:

            # we dynamically create the tool_index
            # if the tool is not in the "tool_index", then we know
            # that it is at index 0
            try:
                index = tool_index[x.tool_no.iloc[1]]
            except:
                tool_index[x.tool_no.iloc[1]] = 0
                index = 0

            name = str(cut_time) + "_{}_{}".format(x.tool_no.iloc[1], index)

            if stable_speed_only == True:
                dict_cuts[str(name)] = stable_speed_section(x.iloc[1:])
                tool_index[x.tool_no.iloc[1]] += 1
            else:
                dict_cuts[str(name)] = x.iloc[1:]
                tool_index[x.tool_no.iloc[1]] += 1
        else:
            pass

    print("Finished split for:\t", str(cut_time))
    return dict_cuts


def low_level_df(
    interim_path,
    features={},
    svd_feature=False,
    fft_features=False,
    list_of_svd_signals=["current_sub", "current_main"],
    svd_feat_count=2,
    svd_window_size=100,
):
    """Create a low-level pandas dataframe with all the cut labels.

    Once the cuts have been split and put into a single dictionary, the
    dictionary should be saved as a pickle file.
    
    Parameters
    ===========
    interim_path : Path object
        Path location that contains the .pickle files. 
        Path such as --> interim_path = Path('/home/user/some_path')

    features : dict
        Dictionary that contains the column name of the feature to be calculated,
        the feature function, and the signal that the function is to be applied to
        -->     {"column_name_1": [feature_function, "signal_name", "spindle_main],
                 "column_name_2": [feature_function, "signal_name", "spindle_sub]"
                }
        e.g.
        features = {"min_current_main":[feat_min_value, "current_main", 'spindle_main'],
                    "max_current_main":[feat_max_value, "current_main", 'spindle_main'],
                    "min_current_sub":[feat_min_value, "current_sub", 'spindle_sub'],
                    "max_current_sub":[feat_max_value, "current_sub", 'spindle_sub'],
                   }
    
    svd_feature : Boolean
        Either True or False. If set as True, the 
    
    Returns
    ===========
    df : pandas dataframe
        Returns a pandas dataframe of the low-level cut information.

    """

    # column names for the dataframe
    columns_low = ["name_low", "unix_date", "date", "tool", "index", "len_cut",] + list(
        features.keys()
    )

    # if svd_feature is True, then create the SVD dictionary
    # (used to create the various SVD column labels)
    if svd_feature:
        svd_feature_dictionary = svd_columns(svd_feat_count, list_of_svd_signals)
    else:
        svd_feature_dictionary = {}

    columns_low = columns_low + list(svd_feature_dictionary.keys())

    # create the dataframe
    df = pd.DataFrame(data=None, columns=columns_low)

    # go through each pickle file (dictionary of the cuts) and extract
    # appropriate info for dataframe.
    row_index = 0
    for file in os.listdir(interim_path):
        if file.endswith(".pickle"):

            pickle_in = open((interim_path / file), "rb")
            df_cut = pickle.load(pickle_in)

            cut_id = file.split(sep=".")[0]  # get the unique name

            j = cut_id.split(sep="_")  # split the cut name at the 'underscore'
            time_unix = int(j[0])  # get the unix time

            # convert unix time into readable time-stamp
            time_stamp = str(
                datetime.fromtimestamp(int(j[0])).strftime("%Y-%m-%d %H:%M:%S")
            )

            tool = int(j[1])  # tool no.
            index = int(j[2])  # index no.
            l_cut = len(df_cut)  # length of cuts

            calculated_features = []

            # determine which spindle is operating, main or sub
            # can also use the 'mean' calculations in the feature engineering functions
            current_main_mean = np.mean(df_cut["current_main"])
            current_sub_mean = np.mean(df_cut["current_sub"])

            if current_main_mean > current_sub_mean:
                spindle_main_running = True
            else:
                spindle_main_running = False

            # calculate the fft, to get yf, if fft_features==True
            if fft_features:
                if spindle_main_running is True:
                    yf, xf = calc_fft(
                        df_cut["current_main"].to_numpy(dtype="float64"), l_cut
                    )
                else:
                    yf, xf = calc_fft(
                        df_cut["current_sub"].to_numpy(dtype="float64"), l_cut
                    )
            else:
                yf = None
                xf = None

            for feat_key in features:
                try:
                    x = features[feat_key][0](
                        df_cut,
                        l_cut,
                        yf,
                        xf,
                        spindle_main_running,
                        features[feat_key][1],
                        features[feat_key][2],
                    )
                    calculated_features.append(x)

                except Exception as ex:
                    logging.exception("Error in code")
                    print("Error with file ", file)
                    calculated_features.append("")

            if svd_feature == True:
                svd_feature_dictionary = svd_columns(
                    svd_feat_count, list_of_svd_signals
                )

                svd_results = feat_svd_values(
                    svd_feature_dictionary, df_cut, svd_window_size, spindle_main_running
                )

                for svd_key in svd_results:
                    calculated_features.append(svd_results[svd_key][2])

            # assign column names to df
            df.loc[row_index] = [
                cut_id,
                time_unix,
                time_stamp,
                tool,
                index,
                l_cut,
            ] + calculated_features

            row_index += 1

            # print out to show how far along we are
            if row_index % 500 == 0:
                print("file no =", row_index)

    return df


def check_date(df_high, df_low):
    """Label each cut in the df_low (low-level cut data) as either failed, or not
    
    Parameters
    ===========
    df_high : pandas dataframe
        Dataframe of the high-level cut information (from a csv file)
        
    df_low : pandas dataframe
        Dataframe of the low-level cut information. This includes all the sub-cut info
        e.g. 1548854748_54_5, 1548854748_54_6, etc.

    Returns
    ===========
    df : pandas dataframe
        Returns a pandas dataframe ofhte low-level cut information with the labels.
    
    """

    # create an empty dictionary to store the failed index in df_low, along with its
    # failure category (e.g. 1 or 2)
    failed_label = {}

    # iterate through each row in the df_high
    for i in df_high.itertuples():
        tool_list = str(
            i.failed_tools
        ).split()  # split the failed tool number on the 'space'

        # iterate through each row in the df_low
        for j in df_low.itertuples():
            # if the date of the cut in the df_high equals that in the df_low
            # and the tool number is in the tool_list, append the failed indicator
            if j.unix_date == i.unix_date and str(j.tool) in tool_list:
                failed_label[j.Index] = str(int(i.failed))
            else:
                pass

    # create a list, equal to length of the df_low, of each cut label
    l = []

    # go through each row in df_low, and apply proper category
    for k in range(len(df_low)):
        if k in list(failed_label.keys()):
            l.append(failed_label[k])
        else:
            l.append("")
    df_low["failed"] = l
    return df_low


def rename_cols_df(df):
    """Take a dataframe with various signals (e.g. current_main, current_sub, etc.)
    and re-lable the columns to standard names
    
    """

    # Dict of columns names to standard names
    col_name_change = {
        "Current_MainSpindle": "current_main",
        "Current_SubSpindle": "current_sub",
        "Power_MainSpindle": "power_main",
        "Power_SubSpindle": "power_sub",
        "CUT_Signal": "cut_signal",
        "Cut_Signal": "cut_signal",
        "Speed_SubSpindle": "speed_sub",
        "Speed_MainSpindle": "speed_main",
        "TOOL_Number": "tool_no",
        "Tool_Number": "tool_no",
        "INORM_MainSpindle": "current_main",
        "INORM_SubSpindle": "current_sub",
        "LMDAT_MainSpindle": "voltage_main",
        "LMDAT_SubSpindle": "voltage_sub",
        "INORM.1": "current_sub",
        "INORM": "current_main",
        "SPSPD": "speed_main",
        "SPSPD.1": "speed_sub",
        "SPEED": "speed_main",
        "SPEED.1": "speed_sub",
        "TCMD": "tcmd_z",
        "TCMD.1": "tcmd_x",
        "ERR": "error_z",
        "PMC": "pmc",
    }

    # Rename columns to standard names
    df.rename(columns=col_name_change, inplace=True)

    return df


def cut_signal_apply(cols):
    """Determine if the tool is in cut, or not, from PMC signal

    
    Explanation
    ===========
    The PMC signal is a binary number with 7 characters (e.g. 1111001, representing 121 
    in base 10 number system). Another example: 0000110 represents 6. If the first digit 
    in the sequence is a 1, then the tool is in cut (1000000 is equal to 64 in base 10). 
    So if tool 6 is in cut, the PMC signal would be 1000110. 1000110 is equivalent to 
    70 in base 10.

    So if the first digit is 1, the tool is in cut. The remaining digits equal the tool number.

    When the PMC signal is saved to the CSV, it is saved as a base 10 number. Work in the base 10 then. 
    Subtract 64 from the number. If the result is greater than 0, then we know the tool is in cut, and the 
    tool number is pmc_no - 64. If, after subtracting 64, the result is negative, we know that the tool 
    is out of cut, and the tool number is equal to pmc_no.

    """
    pmc = cols[0]
    if (pmc - 64) > 0:
        return 1
    else:
        return 0


def tool_no_apply(cols):
    """Gets the tool number from the PMC signal
    
    Explanation
    ===========
    Same explanation as in the cut_signal_apply function

    """

    pmc = cols[0]
    if (pmc - 64) > 0:
        return int(pmc - 64)
    else:
        return int(pmc)
