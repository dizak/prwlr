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
from prowler.errors import *
from prowler.utils import *
from prowler.databases import Columns as _DatabasesColumns
from prowler.profiles import Profile as _Profile


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

def permute_profiles(dataframe,
                     iterations,
                     return_series=False,
                     multiprocessing=False,
                     mp_backend="joblib"):
    """
    Returns list of PSS bins after each permutation.

    The algorithm:
        1. Extract ORFs and PROFs columns.
        2. Make the non-redundant list of ORF-PROF.
        4. Shuffle PROF column using pandas.Series.sample method.
        5. Merge with the stripped DataFrame on ORF (how="left").
        6. Calculate the results.

    Parameters
    -------
    dataframe: pandas.DataFrame
        Dataframe on which test is performed.
    iterations: int
        Number of permutations to perform.
    multiprocessing: bool, default <False>
        pathos multiprocessing is used if <True>. Divides iterations
        between cores.
    """
    def _permute_profiles(dataframe,
                          iteration):
        """
        Returns a interactions network with permuted profiles and re-calculated
        PSS.

        Parameters
        ------
        dataframe: pandas.DataFrame
            Dataframe to be permuted.

        Return
        -------
            pandas.DataFrame
        Dataframe with the profiles permuted among ORFs names and PSS
        re-calculated.
        """
        sub_Q = dataframe[[Columns.ORF_Q,
                           Columns.PROF_Q]].rename(columns={Columns.ORF_Q:
                                                            Columns.ORF,
                                                            Columns.PROF_Q:
                                                            Columns.PROF}).drop_duplicates(subset=[Columns.ORF]).reset_index(drop=True)
        sub_A = dataframe[[Columns.ORF_A,
                           Columns.PROF_A]].rename(columns={Columns.ORF_A:
                                                            Columns.ORF,
                                                            Columns.PROF_A:
                                                            Columns.PROF}).drop_duplicates(subset=[Columns.ORF]).reset_index(drop=True)
        sub_QA = pd.concat([sub_Q,
                            sub_A]).drop_duplicates(subset=[Columns.ORF])
        right_df = pd.concat([sub_QA[Columns.ORF].reset_index(drop=True),
                              sub_QA[Columns.PROF].sample(n=len(sub_QA),
                                                       replace=True).reset_index(drop=True)],
                             axis=1)
        del sub_Q, sub_A, sub_QA
        permuted = pd.merge(left=dataframe.drop([Columns.PROF_Q, Columns.PROF_A, Columns.PSS], axis=1),
                            right=right_df,
                            left_on=[Columns.ORF_Q],
                            right_on=[Columns.ORF],
                            how="left").merge(right_df,
                                              left_on=[Columns.ORF_A],
                                              right_on=[Columns.ORF],
                                              how="left",
                                              suffixes=[Columns.QUERY_SUF, Columns.ARRAY_SUF])
        permuted[Columns.PSS] = permuted.apply(lambda x:
                                            x[Columns.PROF_Q].calculate_pss(x[Columns.PROF_A]),
                                            axis=1)
        del right_df
        gc.collect()
        return pd.DataFrame(permuted.groupby(by=[Columns.PSS]).size())
    if multiprocessing is True:
        f = partial(_permute_profiles, dataframe)
        chunksize = iterations / ptmp.cpu_count()
        out = ptmp.ProcessingPool().map(f, list(range(iterations)), chunksize=chunksize)
    else:
        out = []
        for i in tqdm(list(range(iterations))):
            out.append(_permute_profiles(dataframe, i))
    if return_series:
        return pd.concat([i[1].rename(columns={0: i[0]})
                          for i in enumerate(out)],
                         axis=1).fillna(value=0)
    else:
        return out

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


class Selector(Columns,
               _Profile):
    """
    Allows convenient selections of the Interactions Network.
    """
    def __init__(self,
                 dataframe,
                 profiles_similarity_threshold,
                 p_value=0.05,
                 GIS_min=0.04,
                 GIS_max=-0.04,
                 all_species_in_query=None,
                 any_species_in_query=None,
                 none_species_in_query=None,
                 all_species_in_array=None,
                 any_species_in_array=None,
                 none_species_in_array=None):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Must be pandas.DataFrame")
        if not all([isinstance(i, float) for i in [p_value,
                                                   GIS_min,
                                                   GIS_max]]):
            raise TypeError("Must be float.")
        if not isinstance(profiles_similarity_threshold, int):
            raise TypeError("Must be int.")
        self._summary_dict = {}
        self.dataframe = dataframe
        self._profiles_similarity_threshold = profiles_similarity_threshold
        self._p_value = p_value
        self._GIS_min = GIS_min
        self._GIS_max = GIS_max
        self._all_species_in_query = all_species_in_query
        self._any_species_in_query = any_species_in_query
        self._none_species_in_query = none_species_in_query
        self._all_species_in_array = all_species_in_array
        self._any_species_in_array = any_species_in_array
        self._none_species_in_array = none_species_in_array
        self._summary_dict["total"] = len(self.dataframe)
        try:
            self.positive_DMF = ((self.dataframe[self.DMF] >
                                  self.dataframe[self.SMF_Q]) &
                                 (self.dataframe[self.DMF] >
                                  self.dataframe[self.SMF_A]))
            self.negative_DMF = ((self.dataframe[self.DMF] <
                                  self.dataframe[self.SMF_Q]) &
                                 (self.dataframe[self.DMF] <
                                  self.dataframe[self.SMF_A]))
            self.SMF_below_one = (self.dataframe[self.SMF_Q] < 1.0) &\
                                 (self.dataframe[self.SMF_A] < 1.0)
            self._summary_dict["DMF_positive"] = len(self.dataframe[self.positive_DMF]),
            self._summary_dict["DMF_negative"] = len(self.dataframe[self.negative_DMF]),
        except KeyError:
            warnings.warn("Failed to make fitness-based booleans.",
                          SelectionFailWarning)
        try:
            self.p_value = (self.dataframe[self.GIS_P] <= self._p_value)
        except KeyError:
            warnings.warn("Failed to make p-value-based booleans.",
                          SelectionFailWarning)
        try:
            self.GIS_max = (self.dataframe[self.GIS] < self._GIS_max)
            self.GIS_min = (self.dataframe[self.GIS] > self._GIS_min)
        except KeyError:
            warnings.warn("Failed to make Genetic Interactions Score-based booleans.",
                          SelectionFailWarning)
        try:
            self.PSS_bins = pd.DataFrame(self.dataframe.groupby(by=[self.PSS]).size())
            self.similar_profiles = (self.dataframe["PSS"] >=
                                     self._profiles_similarity_threshold)
            self.dissimilar_profiles = (self.dataframe["PSS"] <=
                                        self._profiles_similarity_threshold)
            self.mirror_profiles = (self.dataframe["PSS"] <=
                                    self._profiles_similarity_threshold)
            self.no_flat_plu_q = (self.dataframe[self.PROF_Q].apply(lambda x: x.to_string()) !=
                                  _Profile._positive_sign * len(self.dataframe.PROF_Q[0]))
            self.no_flat_min_q = (self.dataframe[self.PROF_Q].apply(lambda x: x.to_string()) !=
                                  _Profile._negative_sign * len(self.dataframe.PROF_Q[0]))
            self.no_flat_plu_a = (self.dataframe[self.PROF_A].apply(lambda x: x.to_string()) !=
                                  _Profile._positive_sign * len(self.dataframe.PROF_Q[0]))
            self.no_flat_min_a = (self.dataframe[self.PROF_A].apply(lambda x: x.to_string()) !=
                                  _Profile._negative_sign * len(self.dataframe.PROF_Q[0]))
            self.flat_plu_q = (self.dataframe[self.PROF_Q].apply(lambda x: x.to_string()) ==
                               _Profile._positive_sign * len(self.dataframe.PROF_Q[0]))
            self.flat_min_q = (self.dataframe[self.PROF_Q].apply(lambda x: x.to_string()) ==
                               _Profile._negative_sign * len(self.dataframe.PROF_Q[0]))
            self.flat_plu_a = (self.dataframe[self.PROF_A].apply(lambda x: x.to_string()) ==
                               _Profile._positive_sign * len(self.dataframe.PROF_Q[0]))
            self.flat_min_a = (self.dataframe[self.PROF_A].apply(lambda x: x.to_string()) ==
                               _Profile._negative_sign * len(self.dataframe.PROF_Q[0]))
            self._summary_dict["similar_profiles"] = len(self.dataframe[self.similar_profiles])
            self._summary_dict["dissimilar_profiles"] = len(self.dataframe[self.dissimilar_profiles])
            self._summary_dict["mirror_profiles"] = len(self.dataframe[self.mirror_profiles])
        except KeyError:
            warnings.warn("Failed to make phylogenetic profiles-based booleans",
                          SelectionFailWarning)
        self.summary = pd.DataFrame(self._summary_dict,
                                    index=[0])
        if self._all_species_in_query is not None:
            if not isinstance(self._all_species_in_query, (list, tuple)):
                raise TypeError("Must be list or tuple.")
            try:
                self.all_species_in_query = self.dataframe[self.PROF_Q].apply(lambda x: isiniterable(self._all_species_in_query,
                                                                                                     x.get_present(),
                                                                                                     all_present=True))
            except KeyError:
                warnings.warn("Failed to make query-species-based selection.")
        if self._any_species_in_query is not None:
            if not isinstance(self._any_species_in_query, (list, tuple)):
                raise TypeError("Must be list or tuple.")
            try:
                self.any_species_in_query = self.dataframe[self.PROF_Q].apply(lambda x: isiniterable(self._any_species_in_query,
                                                                                                     x.get_present(),
                                                                                                     all_present=False))
            except KeyError:
                warnings.warn("Failed to make query-species-based selection.")
        if self._none_species_in_query is not None:
            if not isinstance(self._none_species_in_query, (list, tuple)):
                raise TypeError("Must be list or tuple.")
            try:
                self.none_species_in_query = self.dataframe[self.PROF_Q].apply(lambda x: isiniterable(self._none_species_in_query,
                                                                                                      x.get_absent(),
                                                                                                      all_present=True))
            except KeyError:
                warnings.warn("Failed to make query-species-based selection.")
        if self._all_species_in_array is not None:
            if not isinstance(self._all_species_in_array, (list, tuple)):
                raise TypeError("Must be list or tuple.")
            try:
                self.all_species_in_array = self.dataframe[self.PROF_A].apply(lambda x: isiniterable(self._all_species_in_array,
                                                                                                     x.get_present(),
                                                                                                     all_present=True))
            except KeyError:
                warnings.warn("Failed to make array-species-based selection.")
        if self._any_species_in_array is not None:
            if not isinstance(self._any_species_in_array, (list, tuple)):
                raise TypeError("Must be list or tuple.")
            try:
                self.any_species_in_array = self.dataframe[self.PROF_A].apply(lambda x: isiniterable(self._any_species_in_array,
                                                                                                     x.get_present(),
                                                                                                     all_present=False))
            except KeyError:
                warnings.warn("Failed to make array-species-based selection.")
        if self._none_species_in_array is not None:
            if not isinstance(self._none_species_in_array, (list, tuple)):
                raise TypeError("Must be list or tuple.")
            try:
                self.none_species_in_array = self.dataframe[self.PROF_A].apply(lambda x: isiniterable(self._none_species_in_array,
                                                                                                      x.get_absent(),
                                                                                                      all_present=True))
            except KeyError:
                warnings.warn("Failed to make array-species-based selection.")
