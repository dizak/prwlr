# -*- coding: utf-8 -*-


from __future__ import print_function
import gc
import warnings
import pandas as pd
import math
from functools import partial
import numpy as np
import pathos.multiprocessing as ptmp
from tqdm import tqdm
from prwlr.errors import *
from prwlr.utils import *
from prwlr.databases import Columns as _DatabasesColumns
from prwlr.profiles import Profile as _Profile


class Columns(_DatabasesColumns):
    """
    Container for the columns names defined in this module.
    """
    P = "P"
    COUNT = "COUNT"
    COUNT_EXP = "COUNT_EXP"
    SCORE = "SCORE"
    SCORE_EXP = "SCORE_EXP"
    FOLD_CHNG = "FOLD_CHNG"
    SIM = "SIMILAR"
    DIS = "DISSIMILAR"
    MIR = "MIRROR"
    ITER = "ITERATION"
    DATAFRAME = "DATAFRAME"
    dtypes = {SIM: "uint32",
              DIS: "uint32",
              MIR: "uint32",
              ITER: "uint32",
              COUNT: "uint8",
              COUNT_EXP: "float32",
              SCORE: "float32",
              SCORE_EXP: "float32",
              FOLD_CHNG: "float32",
              P: "float32",
              _DatabasesColumns.PSS: _DatabasesColumns.dtypes[_DatabasesColumns.PSS]}


def _log_binomial_coeff(n,
                        k):
    # use multiplicative formula and calculate logarithms on the fly
    n = float(n)
    w = 0.0
    if k > n / 2:    # shorter loop
        k = int(n - k)
    for i in range(1, k + 1):
        w += math.log((n - i + 1.0) / float(i))
    return w

def _score(hit_num,
           prot_num,
           background_p):
    """
    Calculate logarithm of probability that given term was found hit_num times by chance.
    Use binomial distribution:
    log(P) = log((N  k)  * p**k * (1-p)**(N-k)) = log(N k) + k*log(p) + (N-k)*log(1-p)
    """
    prot_num = int(prot_num)
    hit_num = int(hit_num)
    log_p = (_log_binomial_coeff(prot_num, hit_num) +
             hit_num * math.log(background_p) +
             (prot_num - hit_num) * math.log(1.0 - background_p))
    return log_p

def calculate_enrichment(selected,
                         total,
                         col=Columns.PSS):
    """
    Returns enrichment table.

    Parameters
    -------
    selected: pandas.DataFrame
        Dataframe containing data of interest.
    total: pandas.DataFrame
        Dataframe containing all the samples.
    col: str
        Column name holding attribute to calculate enrichment on.

    Returns
    -------
        pandas.DataFrame
    Dataframe with enrichment scores and fold change.
    """
    if len(selected) == 0 or len(total) == 0:
        raise ValueError("selected and total dataframes must not be empty.")
    if len(selected) > len(total):
        raise ValueError("selected must not be longer bigger than total.")
    selected_bins = pd.DataFrame(selected.groupby(by=[col]).size(),
                                 columns=[Columns.COUNT]).reset_index()
    expected_bins = pd.DataFrame(total.groupby(by=[col]).size(),
                                 columns=[Columns.COUNT]).reset_index()
    selected_bins[Columns.P] = expected_bins[Columns.COUNT].apply(lambda x: float(x) / float(len(total)))
    selected_bins[Columns.COUNT_EXP] = selected_bins.apply(lambda x: len(selected) * x[Columns.P],
                                                        axis=1)
    selected_bins[Columns.SCORE] = selected_bins.apply(lambda x: Columns._score(x[Columns.COUNT],
                                                                          len(selected),
                                                                          x[Columns.P]),
                                                    axis=1)
    selected_bins[Columns.SCORE_EXP] = selected_bins.apply(lambda x: Columns._score(x[Columns.COUNT_EXP],
                                                                              len(selected),
                                                                              x[Columns.P]),
                                                        axis=1)
    selected_bins[Columns.FOLD_CHNG] = np.log2(selected_bins[Columns.COUNT] /
                                            selected_bins[Columns.COUNT_EXP])
    selected_bins = selected_bins.astype({k: v for k, v in Columns.dtypes.items()
                                         if k in selected_bins.columns})
    return selected_bins

def binomial_pss_test(desired_pss,
                      selected,
                      total,
                      test_size=1000):
    """
    Draws samples of desired PSS from binomial distribution. Uses
    numpy.random.binomial.

    Parameters
    -------
    desired_pss: int
        Profiles Similarity Score of interest. Interactions with this or
        higher values are considered success.
    selected: pandas.DataFrame
        Dataframe containing data of interest. Interactions with this
        values are subject of the test.
    total: pandas.DataFrame
        Dataframe containing all the samples.

    Returns
    -------
    dict with int values
        <complete> - number of interactions with PSS higher than the
        test single results.
        <average> - average of single test values.
    """
    p = float(len(total[total[Columns.PSS] >= desired_pss])) / float(len(total))
    n = float(len(selected))
    real_val = len(selected[selected[Columns.PSS] >= desired_pss])
    test = np.random.binomial(n, p, test_size)
    return {"complete": sum(test <= real_val),
            "average": sum(test) / len(test)}
