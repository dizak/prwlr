# -*- coding: utf-8 -*-


import gc
import warnings
import pandas as pd
import math
from functools import partial
import numpy as np
import pathos.multiprocessing as ptmp
from tqdm import tqdm
from errors import *
from utils import *
from databases import Columns as _DatabasesColumns
from profiles import Profile as _Profile


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


class Stats(Columns,
            _Profile):
    """
    Calculates and holds data about interactions array statistical
    properties.
    """
    def __init__(self,
                 dataframe,
                 profiles_similarity_threshold,
                 p_value=0.05,
                 GIS_min=0.04,
                 GIS_max=-0.04,
                 query_species_presence_selector=None,
                 query_species_absence_selector=None,
                 array_species_presence_selector=None,
                 array_species_absence_selector=None):
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
        self._query_species_presence_selector = query_species_presence_selector
        self._query_species_absence_selector = query_species_absence_selector
        self._array_species_presence_selector = array_species_presence_selector
        self._array_species_absence_selector = array_species_absence_selector
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
        if self._query_species_presence_selector is not None:
            if not isinstance(self._query_species_presence_selector, (list, tuple)):
                raise TypeError("Must be list or tuple.")
            try:
                self.species_in_query = self.dataframe[self.PROF_Q].apply(lambda x: isiniterable(self._query_species_presence_selector,
                                                                                                 x.get_present(),
                                                                                                 all_present=True))
            except KeyError:
                warnings.warn("Failed to make query-species-based selection.")
        if self._query_species_absence_selector is not None:
            if not isinstance(self._query_species_absence_selector, (list, tuple)):
                raise TypeError("Must be list or tuple.")
            try:
                self.species_not_in_query = self.dataframe[self.PROF_Q].apply(lambda x: isiniterable(self._query_species_absence_selector,
                                                                                                     x.get_absent(),
                                                                                                     all_present=True))
            except KeyError:
                warnings.warn("Failed to make query-species-based selection.")
        if self._array_species_presence_selector is not None:
            if not isinstance(self._array_species_presence_selector, (list, tuple)):
                raise TypeError("Must be list or tuple.")
            try:
                self.species_in_array = self.dataframe[self.PROF_A].apply(lambda x: isiniterable(self._array_species_presence_selector,
                                                                                                 x.get_present(),
                                                                                                 all_present=True))
            except KeyError:
                warnings.warn("Failed to make array-species-based selection.")
        if self._array_species_absence_selector is not None:
            if not isinstance(self._array_species_absence_selector, (list, tuple)):
                raise TypeError("Must be list or tuple.")
            try:
                self.species_not_in_array = self.dataframe[self.PROF_Q].apply(lambda x: isiniterable(self._array_species_absence_selector,
                                                                                                     x.get_absent(),
                                                                                                     all_present=True))
            except KeyError:
                warnings.warn("Failed to make array-species-based selection.")

    def _log_binomial_coeff(self,
                            n,
                            k):
        # use multiplicative formula and calculate logarithms on the fly
        n = float(n)
        w = 0.0
        if k > n / 2:    # shorter loop
            k = int(n - k)

        for i in range(1, k + 1):
            w += math.log((n - i + 1.0) / float(i))
        return w

    def _score(self,
               hit_num,
               prot_num,
               background_p):
        """
        Calculate logarithm of probability that given term was found hit_num times by chance.
        Use binomial distribution:
        log(P) = log((N  k)  * p**k * (1-p)**(N-k)) = log(N k) + k*log(p) + (N-k)*log(1-p)
        """
        prot_num = int(prot_num)
        hit_num = int(hit_num)
        log_p = (self._log_binomial_coeff(prot_num, hit_num) +
                 hit_num * math.log(background_p) +
                 (prot_num - hit_num) * math.log(1.0 - background_p))
        return log_p

    def filter_value(self,
                     dataframe,
                     value):
        """
        Returns interactions network without given values.

        Parameters
        -------
        dataframe: pandas.DataFrame
            Dataframe to be filtered.
        value: any
            Value to be filtered
        """
        for i in dataframe.columns:
            try:
                filtered = dataframe[dataframe[i] == value]
            except TypeError:
                pass
            if filtered.size > 0:
                return filtered

    def calculate_enrichment(self,
                             selected,
                             total,
                             col="PSS"):
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
                                     columns=[self.COUNT]).reset_index()
        expected_bins = pd.DataFrame(total.groupby(by=[col]).size(),
                                     columns=[self.COUNT]).reset_index()
        selected_bins[self.P] = expected_bins[self.COUNT].apply(lambda x: float(x) / float(len(total)))
        selected_bins[self.COUNT_EXP] = selected_bins.apply(lambda x: len(selected) * x[self.P],
                                                            axis=1)
        selected_bins[self.SCORE] = selected_bins.apply(lambda x: self._score(x[self.COUNT],
                                                                              len(selected),
                                                                              x[self.P]),
                                                        axis=1)
        selected_bins[self.SCORE_EXP] = selected_bins.apply(lambda x: self._score(x[self.COUNT_EXP],
                                                                                  len(selected),
                                                                                  x[self.P]),
                                                            axis=1)
        selected_bins[self.FOLD_CHNG] = np.log2(selected_bins[self.COUNT] /
                                                selected_bins[self.COUNT_EXP])
        selected_bins = selected_bins.astype({k: v for k, v in self.dtypes.iteritems()
                                             if k in selected_bins.columns})
        return selected_bins

    def permute_profiles(self,
                         dataframe,
                         iterations,
                         return_series=False,
                         multiprocessing=False):
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
            sub_Q = dataframe[[self.ORF_Q,
                               self.PROF_Q]].rename(columns={self.ORF_Q:
                                                             self.ORF,
                                                             self.PROF_Q:
                                                             self.PROF}).drop_duplicates(subset=[self.ORF]).reset_index(drop=True)
            sub_A = dataframe[[self.ORF_A,
                               self.PROF_A]].rename(columns={self.ORF_A:
                                                             self.ORF,
                                                             self.PROF_A:
                                                             self.PROF}).drop_duplicates(subset=[self.ORF]).reset_index(drop=True)
            sub_QA = pd.concat([sub_Q,
                                sub_A]).drop_duplicates(subset=[self.ORF])
            right_df = pd.concat([sub_QA[self.ORF].reset_index(drop=True),
                                  sub_QA[self.PROF].sample(n=len(sub_QA),
                                                           replace=True).reset_index(drop=True)],
                                 axis=1)
            del sub_Q, sub_A, sub_QA
            permuted = pd.merge(left=dataframe.drop([self.PROF_Q, self.PROF_A, self.PSS], axis=1),
                                right=right_df,
                                left_on=[self.ORF_Q],
                                right_on=[self.ORF],
                                how="left").merge(right_df,
                                                  left_on=[self.ORF_A],
                                                  right_on=[self.ORF],
                                                  how="left",
                                                  suffixes=[self.QUERY_SUF, self.ARRAY_SUF])
            permuted[self.PSS] = permuted.apply(lambda x:
                                                x[self.PROF_Q].calculate_pss(x[self.PROF_A]),
                                                axis=1)
            del right_df
            gc.collect()
            return pd.DataFrame(permuted.groupby(by=[self.PSS]).size())
        if multiprocessing is True:
            f = partial(_permute_profiles, dataframe)
            out = ptmp.ProcessingPool().map(f, range(iterations))
        else:
            out = []
            for i in tqdm(range(iterations)):
                out.append(_permute_profiles(dataframe, i))
        if return_series:
            return pd.concat([i[1].rename(columns={0: i[0]})
                              for i in enumerate(out)],
                             axis=1).fillna(value=0)
        else:
            return out

    def binomial_pss_test(self,
                          desired_pss,
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
        p = float(len(total[total[self.PSS] >= desired_pss])) / float(len(total))
        n = float(len(selected))
        real_val = len(selected[selected[self.PSS] >= desired_pss])
        test = np.random.binomial(n, p, test_size)
        return {"complete": sum(test <= real_val),
                "average": sum(test) / len(test)}
