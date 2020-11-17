import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from sklearn.metrics import (
    roc_auc_score,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
from sklearn.model_selection import ParameterSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from itertools import combinations
import random

# function for selecting a tool in a df, and returning a df with generic names
def make_generic_df(df, tool_no):
    """Function for selecting a tool in a df, and returning a df with generic names"""

    # build the dataframe for the tool in question
    df = df[df["tool"] == tool_no]

    # create the feature list for the main and sub spindle, and generic list
    feat_main = []
    feat_sub = []
    feat_generic = []

    for feat in list(df.columns):
        if "sub" in feat:
            feat_sub.append(feat)
            feat_generic.append(feat.replace("_sub", ""))
        elif "main" in feat:
            feat_main.append(feat)
        else:
            feat_main.append(feat)
            feat_sub.append(feat)
            feat_generic.append(feat)

    # determine which spindle is active for the tool and rename the dataframe with generic columns
    if df["max_current_main"].mean() > df["max_current_sub"].mean():
        df = df[feat_main].set_axis(feat_generic, axis=1, inplace=False)
    else:
        df = df[feat_sub].set_axis(feat_generic, axis=1, inplace=False)


    return df


def get_xy_from_df(
    df,
    tool_list=[54, 36],
    indices_to_keep=[1, 2, 3, 19],
    to_average=True,
    generic_feat_list=None,
):

    """Return X and y data sets from dataframe. These will then be split later"""

    ### TO-DO:
    # * add over-sampling / under-sampling

    # first, merge tool number dataframes into one generic dataframe
    index_count = 0
    for tool in tool_list:
        if index_count == 0:
            df_merge = make_generic_df(df, tool_no=tool)
            index_count += 1
        else:
            df_merge = df_merge.append(
                make_generic_df(df, tool_no=tool), ignore_index=True
            )

    df = df_merge.copy()

    # develop
    if generic_feat_list:
        if "failed" not in generic_feat_list:
            generic_feat_list.append("failed")
    else:
        # create feature list for main and sub spindle
        generic_feat_list = list(df.columns)[6:]

    # if we want to get the average of features across all indices
    if to_average == True:
        index_count = 0
        for i in tool_list:
            if index_count == 0:
                df_merge = (
                    df[(df["tool"] == i) & (df["index"].isin(indices_to_keep))]
                    .groupby(["unix_date"], as_index=False)
                    .mean()
                )

                df_merge['date_ymd'] = df[(df["tool"] == i) & (df["index"].isin(indices_to_keep))].groupby(["unix_date"], as_index=False).first()['date_ymd']     

                index_count += 1

            else:
                df_temp = (
                    df[(df["tool"] == i) & (df["index"].isin(indices_to_keep))]
                    .groupby(["unix_date"], as_index=False)
                    .mean()
                )

                df_temp['date_ymd'] = df[(df["tool"] == i) & (df["index"].isin(indices_to_keep))].groupby(["unix_date"], as_index=False).first()['date_ymd']

                df_merge = df_merge.append(df_temp, ignore_index=True)

    # if we want to keep the individual indices separate
    else:
        df_merge = df[
            (df["tool"].isin(tool_list)) & (df["index"].isin(indices_to_keep))
        ]

    df_merge = df_merge[generic_feat_list + ['date_ymd']].astype({"failed": "int"})
    df_merge = df_merge[
        df_merge["failed"].isin([0, 1])
    ].dropna().reset_index(drop=True)  # only keep 0, 1 failed labels

    # now we will create the X and y data sets
    y = df_merge["failed"].values
    X = df_merge.drop(columns=["failed","date_ymd"]).to_numpy()

    df_ymd_only = df_merge['date_ymd']

    return X, y, df_ymd_only


def under_over_sampler(X, y, method=None, ratio=0.5):
    """Returns an undersampled or oversampled data set. Implemented using imbalanced-learn package.
    ['random_over','random_under','random_under_bootstrap','smote', 'adasyn']
    
    """

    if method == None:
        return X, y

    # oversample methods: https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html
    elif method == "random_over":
        # print('before:',sorted(Counter(y).items()))
        ros = RandomOverSampler(sampling_strategy=ratio, random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        # print('after:',sorted(Counter(y_resampled).items()))
        return X_resampled, y_resampled

    elif method == "random_under":
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=0)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        return X_resampled, y_resampled

    elif method == "random_under_bootstrap":
        rus = RandomUnderSampler(
            sampling_strategy=ratio, random_state=0, replacement=True
        )
        X_resampled, y_resampled = rus.fit_resample(X, y)
        return X_resampled, y_resampled

    elif method == "smote":
        X_resampled, y_resampled = SMOTE(
            sampling_strategy=ratio, random_state=0
        ).fit_resample(X, y)
        return X_resampled, y_resampled

    elif method == "adasyn":
        X_resampled, y_resampled = ADASYN(
            sampling_strategy=ratio, random_state=0
        ).fit_resample(X, y)
        return X_resampled, y_resampled

    else:
        return X, y


def trim_fold_dates(train_folds):
    '''Function that randomly trims 1-2 dates from a fold. This will be used
    to see which "dates" most negatively affect the model performance
    '''
    train_dates_all = [date for sublist in train_folds for date in sublist]

    # randomly remove some dates
    # select the qty to folds to remove dates from (0-3)
    # qty_fold_remove = random.randrange(0, 4)
    qty_fold_remove = 0

    # select the qty of dates to remove from each fold (from 1-2)
    qty_date_remove = 0
    # qty_date_remove = random.randrange(1, 3)

    # shuffle train-folds
    random.shuffle(train_folds)

    train_folds_new = []

    if qty_fold_remove == 0:
        train_folds_new = train_folds
    else:
        for i, fold in enumerate(train_folds):
            if i < qty_fold_remove:

                random.shuffle(fold)
                fold = fold[qty_date_remove:]
                train_folds_new.append(fold)
            else:
                train_folds_new.append(fold)

    train_dates_all_new = [date for sublist in train_folds_new for date in sublist]
    train_dates_removed = list(set(train_dates_all)-set(train_dates_all_new))
    if qty_fold_remove == 0:
        train_dates_removed = ''
    else:
        train_dates_removed = str(train_dates_removed)

    return train_folds, train_dates_removed

def calculate_scores(clf, X_test, y_test,):
    """Helper function for calculating a bunch of scores"""

    y_pred = clf.predict(X_test)

    # need decision function or probability
    # should probably remove the try-except at a later date
    try:
        y_scores = clf.decision_function(X_test)
    except:
        y_scores = clf.predict_proba(X_test)[:, 1]

    n_correct = sum(y_pred == y_test)

    # need to use decision scores, or probabilities, in roc_score
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    
    # calculate the number of threshold values
    # if there is only one threshold value, then we have a straight line in the PR-AUC plot
    # which means the model is erroneous for that k-fold, and thus, we should ignore the model
    no_thresholds = len(thresholds)

    # calculate the precision recall curve and roc_auc curve
    # when to use ROC vs. precision-recall curves, Jason Brownlee http://bit.ly/38vEgnW
    # https://stats.stackexchange.com/questions/113326/what-is-a-good-auc-for-a-precision-recall-curve
    auc_score = auc(recalls, precisions)
    roc_score = roc_auc_score(y_test, y_scores)
    matthews_coef = matthews_corrcoef(y_test, y_pred)


    # calculate precision, recall, f1 scores
    precision_result = precision_score(y_test, y_pred)
    recall_result = recall_score(y_test, y_pred)
    f1_result = f1_score(y_test, y_pred)


    return auc_score, roc_score, matthews_coef, precision_result, recall_result, f1_result, precisions, recalls, thresholds, no_thresholds, y_scores

def classifier_train_manual(
    X_train,
    y_train,
    df_ymd_only,
    train_folds,
    clf,
    scaler_method="standard",
    uo_sample_method=None,
    imbalance_ratio=1,
    train_on_all=False, print_results=False
):

    """Trains a sklearn classifier on a single fold of training data. Returns the ROC_AUC score, with other
    parameters in a pandas df.
    
    
    To-Do:
        - When using SVM, only use under sampling when feature count over a certain size, 
        otherwise will blow up
    """

    train_folds, dates_remove = trim_fold_dates(train_folds)
    
    # below code is modified from 'Hands on Machine Learning' by Geron (pg. 196)
    no_thresholds_list = []
    roc_auc_results = []
    auc_results = []
    matthews_coef_results = []
    precision_results = []
    recall_results = []
    f1_results = []
    precision_recall_scores = []

    if print_results == True:
    # print definitions of precision / recall
        print(
            "\033[1m",
            "Precision:",
            "\033[0m",
            "What proportion of positive identifications were actually correct?",
        )
        print(
            "\033[1m",
            "Recall:",
            "\033[0m",
            "What proportion of actual positives were identified correctly?",
        )



    k_fold_combinations = list(combinations(train_folds,len(train_folds)-1))
    dates_all = [date for sublist in train_folds for date in sublist]

    k_fold_no = float(len(train_folds))

    # implement cross-validation with 
    for i, folds in enumerate(k_fold_combinations):
        # print('#####', i)
        
        # get the train and test dates for each combination
        train_dates = []
        for fold in folds:
            train_dates.extend(fold)
        test_dates = list(set(dates_all)-set(train_dates))
        
        # get the train and test indices
        train_index = df_ymd_only[df_ymd_only.isin(train_dates)].index.values.astype(int)
        test_index = df_ymd_only[df_ymd_only.isin(test_dates)].index.values.astype(int)

        # use clone to do a deep copy of model without copying attached data
        # https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html
        clone_clf = clone(clf)
        X_train_fold = X_train[train_index]
        y_train_fold = y_train[train_index]

        # get the test folds
        X_test_fold = X_train[test_index]
        y_test_fold = y_train[test_index]

        # shuffle data, just in case
        X_train_fold, y_train_fold = shuffle(X_train_fold, y_train_fold, random_state=0)
        X_test_fold, y_test_fold = shuffle(X_test_fold, y_test_fold, random_state=0)

        imb_ratio = np.sum(y_test_fold)/len(y_test_fold)
        print('Imbalance ratio, fold {}: {:.2%}'.format(i, imb_ratio))
        print('No. failures in fold {}: {}'.format(i, np.sum(y_test_fold)))

        # scale the x-train and x-test-fold
        if scaler_method == "standard":
            scaler = StandardScaler()
            scaler.fit(X_train_fold)
            X_train_fold = scaler.transform(X_train_fold)
            X_test_fold = scaler.transform(X_test_fold)
        elif scaler_method == "min_max":
            scaler = MinMaxScaler()
            scaler.fit(X_train_fold)
            X_train_fold = scaler.transform(X_train_fold)
            X_test_fold = scaler.transform(X_test_fold)
        else:
            pass
        

        # add over/under sampling (do this after scaling)
        X_train_fold, y_train_fold = under_over_sampler(
            X_train_fold, y_train_fold, method=uo_sample_method, ratio=imbalance_ratio
        )

        clone_clf.fit(X_train_fold, y_train_fold)

        (
            auc_score,
            roc_score,
            matthews_coef,
            precision_result,
            recall_result,
            f1_result,
            precisions, 
            recalls,
            thresholds,
            no_thresholds,
            y_scores
        ) = calculate_scores(clone_clf, X_test_fold, y_test_fold)

        if i == 0:
            # print('test dates:',test_dates)
            auc_results_min_fold_train = sorted(train_dates)
            auc_results_min_fold_test = sorted(test_dates)
            # print('auc_results_min_fold_test', auc_results_min_fold_test)

        
        elif auc_score < np.min(np.array(auc_results)):
            # these lists will hold the list of dates for the min and max value folds
            # which is useful for understanding why some folds produce poor results
            auc_results_min_fold_train = sorted(train_dates)
            auc_results_min_fold_test = sorted(test_dates)
            # print(auc_results_min_fold_test)

        no_thresholds_list.append(no_thresholds)
        auc_results.append(auc_score)
        precision_results.append(precision_result)
        recall_results.append(recall_result)
        f1_results.append(f1_result)
        roc_auc_results.append(roc_score)
        precision_recall_scores.append([precisions, recalls, thresholds, y_scores,imb_ratio])
        matthews_coef_results.append(matthews_coef)


        if print_results == True:
            print(
                "ROC: {:.3%} \t AUC: {:.3%} \t Pr: {:.3%} \t Re: {:.3%} \t F1: {:.3%}\n".format(
                    roc_score, auc_score, precision_result, recall_result, f1_result
                )
            )

    if print_results == True:
        print("\033[1m", "\nFinal Results:", "\033[0m")
        print(
            "ROC: {:.3%} \t AUC: {:.3%} \t Pr: {:.3%} \t Re: {:.3%} \t F1: {:.3%}".format(
                np.sum(roc_auc_results) / k_fold_no,
                np.sum(auc_results) / k_fold_no,
                np.sum(precision_results) / k_fold_no,
                np.sum(recall_results) / k_fold_no,
                np.sum(f1_results) / k_fold_no,
            )
        )

        # standard deviations
        print(
            "Std: {:.3%} \t Std: {:.3%} \t Std: {:.3%} \t Std: {:.3%} \t Std: {:.3%}".format(
                np.std(roc_auc_results),
                np.std(auc_results),
                np.std(precision_results),
                np.std(recall_results),
                np.std(f1_results),
            )
        )

    result_dict = {

        "train_dates_removed": dates_remove,
        "min_threshold_value": np.min(no_thresholds_list),
        "roc_auc_score": np.sum(roc_auc_results) / k_fold_no,
        "roc_auc_std": np.std(roc_auc_results),
        "roc_auc_min": np.min(roc_auc_results),
        "roc_auc_max": np.max(roc_auc_results),
        "auc_score": np.sum(auc_results) / k_fold_no,
        "auc_std": np.std(auc_results),
        "auc_min": np.min(auc_results),
        "auc_min_fold_train": auc_results_min_fold_train,
        "auc_min_fold_test": auc_results_min_fold_test,
        "auc_max": np.max(auc_results),
        "matthews_coef": np.sum(matthews_coef_results) / k_fold_no,
        "matthews_std": np.std(matthews_coef_results),
        "matthews_min": np.min(matthews_coef_results),
        "matthews_max": np.max(matthews_coef_results),
        "f1_score": np.sum(f1_results) / k_fold_no,
        "f1_std": np.std(f1_results),
        "f1_min": np.min(f1_results),
        "f1_max": np.max(f1_results),
        "precision": np.sum(precision_results) / k_fold_no,
        "precision_std": np.std(precision_results),
        "precision_min": np.min(precision_results),
        "precision_max": np.max(precision_results),
        "recall": np.sum(recall_results) / k_fold_no,
        "recall_std": np.std(recall_results),
        "recall_min": np.min(recall_results),
        "recall_max": np.max(recall_results),

    }

    # when to use ROC vs. precision-recall curves, Jason Brownlee http://bit.ly/38vEgnW
    # https://stats.stackexchange.com/questions/113326/what-is-a-good-auc-for-a-precision-recall-curve

    if train_on_all == True:
        # now scale and fit the data on the entire training set
        new_clf = clone(clf)

        # scale the x-train and x-test-fold
        if scaler_method == "standard":
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)

        elif scaler_method == "min_max":
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)

        else:
            pass

        X_train, y_train = under_over_sampler(
            X_train, y_train, method=uo_sample_method, ratio=imbalance_ratio
        )
        # print("Training on All Data")

        new_clf.fit(X_train, y_train)

        return result_dict, scaler, new_clf, precision_recall_scores

    else:

        return result_dict, scaler, "", precision_recall_scores



def classifier_train(
    X_train,
    y_train,
    clf,
    scaler_method="standard",
    uo_sample_method=None,
    imbalance_ratio=1,
    train_on_all=False, print_results=False
):

    """Trains a sklearn classifier using k-fold cross-validation. Returns the ROC_AUC score, with other
    parameters in a pandas df.
    
    
    To-Do:
        - When using SVM, only use under sampling when feature count over a certain size, 
        otherwise will blow up
    """


    skfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # below code is modified from 'Hands on Machine Learning' by Geron (pg. 196)
    roc_auc_results = []
    auc_results = []
    precision_results = []
    recall_results = []
    f1_results = []

    if print_results == True:
    # print definitions of precision / recall
        print(
            "\033[1m",
            "Precision:",
            "\033[0m",
            "What proportion of positive identifications were actually correct?",
        )
        print(
            "\033[1m",
            "Recall:",
            "\033[0m",
            "What proportion of actual positives were identified correctly?",
        )

    # implement cross-validation with 
    for train_index, test_index in skfolds.split(X_train, y_train):
        # use clone to do a deep copy of model without copying attached data
        # https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html
        clone_clf = clone(clf)
        X_train_fold = X_train[train_index]
        y_train_fold = y_train[train_index]

        # get the test folds
        X_test_fold = X_train[test_index]
        y_test_fold = y_train[test_index]

        # scale the x-train and x-test-fold
        if scaler_method == "standard":
            scaler = StandardScaler()
            scaler.fit(X_train_fold)
            X_train_fold = scaler.transform(X_train_fold)
            X_test_fold = scaler.transform(X_test_fold)
        elif scaler_method == "min_max":
            scaler = MinMaxScaler()
            scaler.fit(X_train_fold)
            X_train_fold = scaler.transform(X_train_fold)
            X_test_fold = scaler.transform(X_test_fold)
        else:
            pass
        

        # add over/under sampling (do this after scaling)
        X_train_fold, y_train_fold = under_over_sampler(
            X_train_fold, y_train_fold, method=uo_sample_method, ratio=imbalance_ratio
        )

        clone_clf.fit(X_train_fold, y_train_fold)

        (
            auc_score,
            roc_score,
            precision_result,
            recall_result,
            f1_result,
        ) = calculate_scores(clone_clf, X_test_fold, y_test_fold)

        auc_results.append(auc_score)
        precision_results.append(precision_result)
        recall_results.append(recall_result)
        f1_results.append(f1_result)
        roc_auc_results.append(roc_score)

        if print_results == True:
            print(
                "ROC: {:.3%} \t AUC: {:.3%} \t Pr: {:.3%} \t Re: {:.3%} \t F1: {:.3%}".format(
                    roc_score, auc_score, precision_result, recall_result, f1_result
                )
            )

    if print_results == True:
        print("\033[1m", "\nFinal Results:", "\033[0m")
        print(
            "ROC: {:.3%} \t AUC: {:.3%} \t Pr: {:.3%} \t Re: {:.3%} \t F1: {:.3%}".format(
                np.sum(roc_auc_results) / 10.0,
                np.sum(auc_results) / 10.0,
                np.sum(precision_results) / 10.0,
                np.sum(recall_results) / 10.0,
                np.sum(f1_results) / 10.0,
            )
        )

        # standard deviations
        print(
            "Std: {:.3%} \t Std: {:.3%} \t Std: {:.3%} \t Std: {:.3%} \t Std: {:.3%}".format(
                np.std(roc_auc_results),
                np.std(auc_results),
                np.std(precision_results),
                np.std(recall_results),
                np.std(f1_results),
            )
        )

    result_dict = {

        "roc_auc_score": np.sum(roc_auc_results) / 10.0,
        "roc_auc_std": np.std(roc_auc_results),
        "auc_score": np.sum(auc_results) / 10.0,
        "auc_std": np.std(auc_results),
        "f1_score": np.sum(f1_results) / 10.0,
        "f1_std": np.std(f1_results),
        "precision": np.sum(precision_results) / 10.0,
        "precision_std": np.std(precision_results),
        "recall": np.sum(recall_results) / 10.0,
        "recall_std": np.std(f1_results),

    }

    # when to use ROC vs. precision-recall curves, Jason Brownlee http://bit.ly/38vEgnW
    # https://stats.stackexchange.com/questions/113326/what-is-a-good-auc-for-a-precision-recall-curve

    if train_on_all == True:
        # now scale and fit the data on the entire training set
        new_clf = clone(clf)

        # scale the x-train and x-test-fold
        if scaler_method == "standard":
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)

        elif scaler_method == "min_max":
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)

        else:
            pass

        X_train, y_train = under_over_sampler(
            X_train, y_train, method=uo_sample_method, ratio=imbalance_ratio
        )
        # print("Training on All Data")

        new_clf.fit(X_train, y_train)

        return result_dict, scaler, new_clf

    else:

        return result_dict, scaler, ""


def plot_precision_recall_vs_threshold(
    precisions, recalls, thresholds, precision_threshold_setting=0.8
):
    """From Aurélien Geron ML-2, 
    https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb"""

    def plotter_make(precisions, recalls, thresholds):
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
        plt.legend(fontsize=16)
        plt.xlabel("Threshold", fontsize=16)
        plt.grid(True)
        plt.axis([np.min(thresholds), np.max(thresholds), 0, 1])

    recall_90_precision = recalls[np.argmax(precisions >= precision_threshold_setting)]
    threshold_90_precision = thresholds[
        np.argmax(precisions >= precision_threshold_setting)
    ]

    plt.figure(figsize=(8, 4))
    plotter_make(precisions, recalls, thresholds)
    plt.plot(
        [threshold_90_precision, threshold_90_precision],
        [0.0, precision_threshold_setting],
        "r:",
    )
    plt.plot(
        [np.min(thresholds), threshold_90_precision],
        [precision_threshold_setting, precision_threshold_setting],
        "r:",
    )
    plt.plot(
        [np.min(thresholds), threshold_90_precision],
        [recall_90_precision, recall_90_precision],
        "r:",
    )
    plt.plot([threshold_90_precision], [precision_threshold_setting], "ro")
    plt.plot([threshold_90_precision], [recall_90_precision], "ro")
    plt.show()


def plot_precision_vs_recall(precisions, recalls):
    """From Aurélien Geron ML-2, 
    https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb"""

    def plotter_make(precisions, recalls):
        plt.plot(recalls, precisions, "b-", linewidth=2)
        plt.xlabel("Recall", fontsize=16)
        plt.ylabel("Precision", fontsize=16)
        plt.axis([0, 1, 0, 1])
        plt.grid(True)

    plt.figure(figsize=(8, 6))
    plotter_make(precisions, recalls)
    # plt.plot([0.4368, 0.4368], [0., 0.9], "r:")
    # plt.plot([0.0, 0.4368], [0.9, 0.9], "r:")
    # plt.plot([0.4368], [0.9], "ro")
    plt.show()

