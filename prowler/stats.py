# -*- coding: utf-8 -*-


import pandas as pd
import math
import numpy as np
import pathos.multiprocessing as ptmp
from tqdm import tqdm


class Stats:
    """
    Calculates and holds data about interactions array statistical
    properties.
    """
    def __init__(self,
                 inter_df,
                 p_value=0.05,
                 inter_score_min=0.04,
                 inter_score_max=-0.04):
        self.inter_df = inter_df
        self.filters_used = "No filters"
        self.filters_name = "no_filters"
        self.inter_score_min = inter_score_min
        self.inter_score_max = inter_score_max
        self.p_value = (self.inter_df["GIS_P"] <= p_value)
        self.positive_DMF_bool = ((self.inter_df["DMF"] >
                                   self.inter_df["SMF_Q"]) &
                                  (self.inter_df["DMF"] >
                                   self.inter_df["SMF_A"]))
        self.negative_DMF_bool = ((self.inter_df["DMF"] <
                                   self.inter_df["SMF_Q"]) &
                                  (self.inter_df["DMF"] <
                                   self.inter_df["SMF_A"]))
        self.SMF_below_one_bool = (self.inter_df["SMF_Q"] < 1.0) &\
                                  (self.inter_df["SMF_A"] < 1.0)


class Ortho_Stats:
    """Calculates and holds data about interactions array statistical
    properties.
    """
    def __init__(self,
                 query_specices,
                 gene_profiles,
                 inter_df):
        self.query_species = query_specices
        self.inter_df = inter_df
        self.gene_profiles = gene_profiles
        self.num_prop_res = None
        self.e_value = None
        self.perm_results = None
        self.prof_arr_perm_res_avg = None
        self.filters_used = "No filters"
        self.filters_name = "no_filters"

    def df_selector(self,
                    DMF=None,
                    SMF_below_one=True,
                    inter_score_min=None,
                    inter_score_max=None,
                    no_flat_plus=False,
                    no_flat_minus=False,
                    process=None,
                    profiles=None,
                    prof_sim_lev=None):
        """Return filtered Ortho_Stats.interact_df_stats passed from
        Ortho_Interactions (pandas.DataFrame). For each filter, type <None> to
        omit.

        Args:
            DMF (str): selects DMF type. Possible: <positive>, <negative> or
            <None> (omit filter). Default: <None>
            inter_score_min (float): selects minimum Genetic interactions Score.
            Default <None>
            inter_score_max (float): selects minimum Genetic interactions Score.
            Default <None>
            no_flat (bool): eliminates mirror profiles. Default <False>
            process (str): selects bioprocesses similarity. Possible: <identical>,
            "different" or <None>. Default <None>
            profiles (str): selects similar or dissimilar profiles. Possible
            <similar>, <unsimilar> or <None>. Similarity threshold MUST be
            specified with the prof_sim_lev arg if profiles != <None>.
            Default <None>.
            prof_sim_lev (int): defines profiles as similar of dissimilar
            when above or below this given value
        """
        if profiles is None and prof_sim_lev is not None:
            raise ValueError("No value for profiles")
        elif profiles is not None and prof_sim_lev is None:
            raise ValueError("No value for prof_sim_lev")
        else:
            pass
        self.filters_used = []
        self.filters_name = []
        positive_DMF_bool = ((self.inter_df["DMF"] >
                             self.inter_df["SMF_Q"]) &
                             (self.inter_df["DMF"] >
                              self.inter_df["SMF_A"]))
        negative_DMF_bool = ((self.inter_df["DMF"] <
                             self.inter_df["SMF_Q"]) &
                             (self.inter_df["DMF"] <
                              self.inter_df["SMF_A"]))
        SMF_below_one_bool = (self.inter_df["SMF_Q"] < 1.0) &\
                             (self.inter_df["SMF_A"] < 1.0)
        inter_score_max_bool = (self.inter_df["GIS"] < inter_score_max)
        inter_score_min_bool = (self.inter_df["GIS"] > inter_score_min)
        no_flat_plu_q_bool = (self.inter_df["PROF_Q"] !=
                              "+" * len(self.query_species))
        no_flat_min_q_bool = (self.inter_df["PROF_Q"] !=
                              "-" * len(self.query_species))
        no_flat_plu_a_bool = (self.inter_df["PROF_A"] !=
                              "+" * len(self.query_species))
        no_flat_min_a_bool = (self.inter_df["PROF_A"] !=
                              "-" * len(self.query_species))
        iden_proc_bool = (self.inter_df["BSS"] ==
                          "identical")
        diff_proc_bool = (self.inter_df["BSS"] ==
                          "different")
        if profiles is not None:
            sim_prof_bool = (self.inter_df["PSS"] >=
                             prof_sim_lev)
            unsim_prof_bool = (self.inter_df["PSS"] <
                               prof_sim_lev)
        else:
            pass
        if DMF == "positive":
            self.inter_df = self.inter_df[positive_DMF_bool]
            self.filters_used.append("DMF positive")
            self.filters_name.append("DMF_p")
        elif DMF == "negative":
            self.inter_df = self.inter_df[negative_DMF_bool]
            self.filters_used.append("DMF negative")
            self.filters_name.append("DMF_n")
        else:
            pass
        if SMF_below_one is True:
            self.inter_df = self.inter_df[SMF_below_one_bool]
            self.filters_used.append("SMF < 1.0")
            self.filters_name.append("SMF_blw_1")
        else:
            pass
        if isinstance(inter_score_max, float) is True:
            self.inter_df = self.inter_df[inter_score_max_bool]
            self.filters_used.append("Genetic interaction score < {0}".format(inter_score_max))
            self.filters_name.append("gis_{0}".format(inter_score_max))
        else:
            pass
        if isinstance(inter_score_min, float) is True:
            self.inter_df = self.inter_df[inter_score_min_bool]
            self.filters_used.append("Genetic interaction score > {0}".format(inter_score_min))
            self.filters_name.append("gis_{0}".format(inter_score_min))
        else:
            pass
        if no_flat_plus is True:
            self.inter_df = self.inter_df[no_flat_plu_q_bool]
            self.inter_df = self.inter_df[no_flat_plu_a_bool]
            self.filters_used.append("No plus-only (eg ++++++) profiles")
            self.filters_name.append("no_plus_flat")
        else:
            pass
        if no_flat_minus is True:
            self.inter_df = self.inter_df[no_flat_min_q_bool]
            self.inter_df = self.inter_df[no_flat_min_a_bool]
            self.filters_used.append("No minus-only (eg ------) profiles")
            self.filters_name.append("no_min_flat")
        else:
            pass
        if process == "identical":
            self.inter_df = self.inter_df[iden_proc_bool]
            self.filters_used.append("Identical bioprocesses")
            self.filters_name.append("iden_proc")
        elif process == "different":
            self.inter_df = self.inter_df[diff_proc_bool]
            self.filters_used.append("Different bioprocesses")
            self.filters_name.append("diff_proc")
        else:
            pass
        if profiles == "similar":
            self.inter_df = self.inter_df[sim_prof_bool]
            self.filters_used.append("Similar profiles")
            self.filters_name.append("sim_prof")
        elif profiles == "unsimilar":
            self.inter_df = self.inter_df[unsim_prof_bool]
            self.filters_used.append("Dissimilar profiles")
            self.filters_name.append("dis_prof")
        else:
            pass

    def df_num_prop(self,
                    in_prof_sim_lev=None):
        """Return Ortho_Stats.tot_inter_num (int),
        Ortho_Stats.DMF_positive_num (int),
        Ortho_Stats.DMF_negative_num (int),
        Ortho_Stats.sim_prof_num (int).

        Args:
            in_prof_sim_lev (int): defines minimal Genome.gene_profiles in
            Ortho_Stats.inter_df similarity treshold. Set to <None> to
            omit, eg when dispalying multiple thresholds at once.
            Default: <None>
        """
        if isinstance(self.filters_used, str) is True:
            self.filters_used = []
        else:
            pass
        if isinstance(self.filters_name, str) is True:
            self.filters_name = []
        else:
            pass
        positive_DMF_bool = ((self.inter_df["DMF"] >
                             self.inter_df["SMF_Q"]) &
                             (self.inter_df["DMF"] >
                              self.inter_df["SMF_A"]))
        negative_DMF_bool = ((self.inter_df["DMF"] <
                             self.inter_df["SMF_Q"]) &
                             (self.inter_df["DMF"] <
                              self.inter_df["SMF_A"]))
        sim_prof_bool = (self.inter_df["PSS"] >=
                         in_prof_sim_lev)
        unsim_prof_bool = (self.inter_df["PSS"] <
                           in_prof_sim_lev) &\
                          (self.inter_df["PSS"] > 0)
        mir_prof_bool = (self.inter_df["PSS"] == 0)
        if in_prof_sim_lev is None:
            self.num_prop_res = pd.Series({"total": len(self.inter_df),
                                           "DMF_positive": len(self.inter_df[positive_DMF_bool]),
                                           "DMF_negative": len(self.inter_df[negative_DMF_bool]),
                                           "histogram_bins": pd.value_counts(self.inter_df["PSS"])})
        else:
            self.num_prop_res = pd.Series({"total": len(self.inter_df),
                                           "DMF_positive": len(self.inter_df[positive_DMF_bool]),
                                           "DMF_negative": len(self.inter_df[negative_DMF_bool]),
                                           "similar_profiles": len(self.inter_df[sim_prof_bool]),
                                           "unsimilar_profiles": len(self.inter_df[unsim_prof_bool]),
                                           "mirror_profiles": len(self.inter_df[mir_prof_bool]),
                                           "histogram_bins": pd.value_counts(self.inter_df["PSS"])})
            self.filters_used.append("Profiles similarity threshold: {0}".format(in_prof_sim_lev))
            self.filters_name.append("prof_sim_th_{0}".format(in_prof_sim_lev))

    def names_perm(self,
                   e_value,
                   in_prof_sim_lev):
        """Return pandas.DataFrame of number of different types of profiles scores, each
        generated from pandas DataFrame in which genes names were permuted. It is
        an equivalent of creating completely new, random network.

        Args:
            e_value (int): number of times to shuffle the pandas DataFrame
            in_prof_sim_lev(int): treshold for assuming profiles as similar or
            not
        """
        def f(in_iter):
            temp_score_list = []
            q_temp_df = self.inter_df[["GENE_Q", "PROF_Q"]]
            a_temp_df = self.inter_df[["GENE_A", "PROF_A"]]
            q_temp_perm_df = q_temp_df.sample(len(q_temp_df))
            a_temp_perm_df = a_temp_df.sample(len(a_temp_df))
            q_temp_perm_df.index = range(len(q_temp_perm_df))
            a_temp_perm_df.index = range(len(a_temp_perm_df))
            qa_temp_perm_df = pd.concat([q_temp_perm_df, a_temp_perm_df], axis=1)
            for ii in qa_temp_perm_df.itertuples():
                temp_score_list.append(df_qa_names_2_prof_score([getattr(ii, "GENE_Q"),
                                                                 getattr(ii, "GENE_A")],
                                                                self.gene_profiles))
            temp_score_df = pd.DataFrame(temp_score_list,
                                         index=qa_temp_perm_df.index,
                                         columns=["PSS"])
            qa_temp_perm_score_df = pd.concat([qa_temp_perm_df, temp_score_df],
                                              axis=1)
            sim_prof_bool = (qa_temp_perm_score_df["PSS"] >=
                             in_prof_sim_lev)
            unsim_prof_bool = (qa_temp_perm_score_df["PSS"] <
                               in_prof_sim_lev) &\
                              (qa_temp_perm_score_df["PSS"] > 0)
            mir_prof_bool = (qa_temp_perm_score_df["PSS"] == 0)
            sim_prof_perm_num = len(qa_temp_perm_score_df[sim_prof_bool])
            unsim_prof_perm_num = len(qa_temp_perm_score_df[unsim_prof_bool])
            mir_prof_perm_num = len(qa_temp_perm_score_df[mir_prof_bool])
            return {"similar": sim_prof_perm_num,
                    "unsimilar": unsim_prof_perm_num,
                    "mirror": mir_prof_perm_num,
                    "iteration": in_iter + 1}
        perm_results_temp_dict = ptmp.ProcessingPool().map(f, range(e_value))
        self.perm_results = pd.DataFrame(perm_results_temp_dict)

    def prof_cols_perm(self,
                       e_value,
                       in_prof_sim_lev):
        """Return pandas.DataFrame of number of different types of profiles scores,
        ech generated from pandas.DataFrame in which gene profiles were permuted but
        NOT the rest of the data. It is an equivalent of permuting parameters in
        the interactions network without changing the network's topology. Gene
        profiles are shuffled without any key.

        Args:
            e_value (int): number of times to shuffle the pandas DataFrame
            in_prof_sim_lev(int): treshold for assuming profiles as similar or
            not
        """
        q_sign_per_col_profs_cols = ["{0}_Q".format(i) for i in self.query_species]
        a_sign_per_col_profs_cols = ["{0}_A".format(i) for i in self.query_species]

        def f(in_iter):
            temp_score_list = []
            q_prof_temp_df = self.inter_df["PROF_Q"]
            a_prof_temp_df = self.inter_df["PROF_A"]
            drop_prof_temp_df = self.inter_df.drop(["PROF_Q",
                                                    "PROF_A",
                                                    "PSS"] +
                                                   q_sign_per_col_profs_cols +
                                                   a_sign_per_col_profs_cols,
                                                   axis=1)
            q_prof_perm_temp_df = q_prof_temp_df.sample(len(q_prof_temp_df))
            a_prof_perm_temp_df = a_prof_temp_df.sample(len(a_prof_temp_df))
            q_prof_perm_temp_df.index = drop_prof_temp_df.index
            a_prof_perm_temp_df.index = drop_prof_temp_df.index
            permuted_df = pd.concat([drop_prof_temp_df,
                                     q_prof_perm_temp_df,
                                     a_prof_perm_temp_df],
                                    axis=1)
            for ii in permuted_df.itertuples():
                temp_score_list.append([simple_profiles_scorer(np.array(list(getattr(ii, "PROF_Q"))),
                                                               np.array(list(getattr(ii, "PROF_A"))))])
            temp_score_df = pd.DataFrame(temp_score_list,
                                         index=permuted_df.index,
                                         columns=["PSS"])
            permuted_profs_df = pd.concat([permuted_df,
                                          temp_score_df],
                                          axis=1)
            sim_prof_bool = (permuted_profs_df["PSS"] >=
                             in_prof_sim_lev)
            unsim_prof_bool = (permuted_profs_df["PSS"] <
                               in_prof_sim_lev) &\
                              (permuted_profs_df["PSS"] > 0)
            mir_prof_bool = (permuted_profs_df["PSS"] == 0)
            sim_prof_perm_num = len(permuted_profs_df[sim_prof_bool])
            unsim_prof_perm_num = len(permuted_profs_df[unsim_prof_bool])
            mir_prof_perm_num = len(permuted_profs_df[mir_prof_bool])
            return {"similar": sim_prof_perm_num,
                    "unsimilar": unsim_prof_perm_num,
                    "mirror": mir_prof_perm_num,
                    "iteration": in_iter + 1}
        perm_results_temp_dict = ptmp.ProcessingPool().map(f, range(e_value))
        self.perm_results = pd.DataFrame(perm_results_temp_dict)

    def prof_arr_perm(self,
                      e_value,
                      in_prof_sim_lev):
        """Return a new Ortho_Stats.inter_df which was stripped from
        gene_profiles data and then appended with gene_profiles again using
        a permuted gene_profiles list.

        Args:
            e_value (int): number of times to shuffle the pandas DataFrame
            in_prof_sim_lev(int): treshold for assuming profiles as similar or
            not
        """
        q_sign_per_col_profs_cols = ["{0}_Q".format(i) for i in self.query_species]
        a_sign_per_col_profs_cols = ["{0}_A".format(i) for i in self.query_species]
        drop_prof_temp_df = self.inter_df.drop(["PROF_Q",
                                                "PROF_A",
                                                "PSS"] +
                                               q_sign_per_col_profs_cols +
                                               a_sign_per_col_profs_cols,
                                               axis=1)

        def f(in_iter):
            gene_profs_perm_arr_list = []
            prof_score_temp_list = []
            qa_attrib_temp_list = []
            conc_qa_prof_temp_list = []
            gene_profs_names = [i[0] for i in self.gene_profiles]
            gene_profs_profs = [i[1:] for i in self.gene_profiles]
            gene_profs_names_ser = pd.Series(gene_profs_names)
            gene_profs_profs_ser = pd.Series(gene_profs_profs)
            gene_profs_names_ser_perm = gene_profs_names_ser.sample(len(gene_profs_names_ser))
            gene_profs_names_ser_perm.index = range(len(gene_profs_names_ser_perm))
            gene_profs_perm_df = pd.concat([gene_profs_names_ser_perm,
                                            gene_profs_profs_ser],
                                           axis=1)
            gene_profs_perm_df.columns = ["perm_names", "profiles"]
            for i in gene_profs_perm_df.itertuples():
                name_arr = np.array(getattr(i, "perm_names"))
                full_arr = np.append(name_arr, getattr(i, "profiles"))
                gene_profs_perm_arr_list.append(full_arr)
            for i in drop_prof_temp_df.itertuples():
                qa_attrib_temp_list.append([getattr(i, "GENE_Q"),
                                            getattr(i, "GENE_A")])
            for i in qa_attrib_temp_list:
                prof_score_temp_list.append(df_qa_names_2_prof_score(i,
                                                                     gene_profs_perm_arr_list))
                conc_qa_prof_temp_list.append([gene_profile_finder_by_name(i[0],
                                                                           gene_profs_perm_arr_list,
                                                                           conc=True),
                                               gene_profile_finder_by_name(i[1],
                                                                           gene_profs_perm_arr_list,
                                                                           conc=True)])
            prof_score_temp_df = pd.DataFrame(prof_score_temp_list,
                                              index=drop_prof_temp_df.index,
                                              columns=["PSS"])
            profs_pairs_temp_df = pd.DataFrame(conc_qa_prof_temp_list,
                                               index=drop_prof_temp_df.index,
                                               columns=["PROF_Q", "PROF_A"])
            permuted_df = pd.concat([drop_prof_temp_df,
                                     profs_pairs_temp_df,
                                     prof_score_temp_df],
                                    axis=1)
            sim_prof_bool = (permuted_df["PSS"] >=
                             in_prof_sim_lev)
            unsim_prof_bool = (permuted_df["PSS"] <
                               in_prof_sim_lev) &\
                              (permuted_df["PSS"] > 0)
            mir_prof_bool = (permuted_df["PSS"] == 0)
            sim_prof_perm_num = len(permuted_df[sim_prof_bool])
            unsim_prof_perm_num = len(permuted_df[unsim_prof_bool])
            mir_prof_perm_num = len(permuted_df[mir_prof_bool])
            return {"similar": sim_prof_perm_num,
                    "unsimilar": unsim_prof_perm_num,
                    "mirror": mir_prof_perm_num,
                    "iteration": in_iter + 1,
                    "dataframe": permuted_df}
        permuted_df_results_temp = ptmp.ProcessingPool().map(f, range(e_value))
        self.prof_arr_perm_results = pd.DataFrame(permuted_df_results_temp)
        self.prof_arr_perm_res_avg = pd.Series({"mirror_profiles": sum(self.prof_arr_perm_results.mirror) / len(self.prof_arr_perm_results),
                                                "similar_profiles": sum(self.prof_arr_perm_results.similar) / len(self.prof_arr_perm_results),
                                                "unsimilar": sum(self.prof_arr_perm_results.unsimilar) / len(self.prof_arr_perm_results)})

    def KO_profs_perm(self,
                      e_value,
                      in_prof_sim_lev):
        """Return Ortho_Stats.prof_arr_perm_results pandas.DataFrame containing
        number of similar, dissimilar, mirror profiles and complete permuted
        pandas.DataFrame itself. Return Ortho_Stats.prof_arr_perm_res_avg
        containing average numbers of similar, dissimilar and mirror profiles.
        The algorithm:
            1. Extract Ortho_Stats.inter_df["ORF", "PROF"].
            2. Strip the original DataFrame from these 2 cols.
            3. Make the non-redundant list.
            4. Shuffle PROF col using pandas.Series.sample method.
            5. Merge with the stripped DataFrame on ORF (how="left").
            6. Calculate the results.

        Args:
            e_value (int): number of times to shuffle the pandas DataFrame
            in_prof_sim_lev(int): treshold for assuming profiles as similar or
            not
        """
        def f(in_iter):
            q_ORF_prof_df = self.inter_df[["ORF_Q",
                                           "PROF_Q"]]
            a_ORF_prof_df = self.inter_df[["ORF_A",
                                           "PROF_A"]]
            drop_prof_temp_df = self.inter_df.drop(["PROF_Q",
                                                    "PROF_A",
                                                    "PSS"],
                                                   axis=1)
            q_ORF_prof_df.columns = range(len(q_ORF_prof_df.columns))
            a_ORF_prof_df.columns = range(len(a_ORF_prof_df.columns))
            stack_ORF_prof_df = pd.concat([q_ORF_prof_df,
                                           a_ORF_prof_df],
                                          ignore_index=True)
            stack_ORF_prof_df.drop_duplicates(inplace=True)
            stack_ORF_prof_df.columns = ["ORF", "PROF"]
            stack_ORF_prof_df.index = range(len(stack_ORF_prof_df))
            stack_prof_perm_df = stack_ORF_prof_df.PROF.sample(len(stack_ORF_prof_df))
            stack_prof_perm_df.index = range(len(stack_prof_perm_df))
            ORF_prof_perm_df = pd.concat([stack_ORF_prof_df.ORF,
                                          stack_prof_perm_df],
                                         axis=1)
            q_merged_df = pd.merge(drop_prof_temp_df,
                                   ORF_prof_perm_df,
                                   left_on="ORF_Q",
                                   right_on="ORF",
                                   how="left")
            qa_merged_df = pd.merge(q_merged_df,
                                    ORF_prof_perm_df,
                                    left_on="ORF_A",
                                    right_on="ORF",
                                    how="left",
                                    suffixes=("_Q", "_A"))
            qa_merged_score_df = df_based_profiles_scorer(qa_merged_df,
                                                          prof_1_col_name="PROF_Q",
                                                          prof_2_col_name="PROF_A",
                                                          score_col_name="PSS")
            sim_prof_bool = (qa_merged_score_df["PSS"] >=
                             in_prof_sim_lev)
            unsim_prof_bool = (qa_merged_score_df["PSS"] <
                               in_prof_sim_lev) &\
                              (qa_merged_score_df["PSS"] > 0)
            mir_prof_bool = (qa_merged_score_df["PSS"] == 0)
            sim_prof_perm_num = len(qa_merged_score_df[sim_prof_bool])
            unsim_prof_perm_num = len(qa_merged_score_df[unsim_prof_bool])
            mir_prof_perm_num = len(qa_merged_score_df[mir_prof_bool])
            return {"similar": sim_prof_perm_num,
                    "unsimilar": unsim_prof_perm_num,
                    "mirror": mir_prof_perm_num,
                    "iteration": in_iter + 1,
                    "dataframe": qa_merged_score_df}
        permuted_df_results_temp = ptmp.ProcessingPool().map(f, range(e_value))
        self.prof_KO_perm_results = pd.DataFrame(permuted_df_results_temp)
        self.prof_KO_perm_res_avg = pd.Series({"mirror_profiles": sum(self.prof_KO_perm_results.mirror) /
                                              len(self.prof_KO_perm_results),
                                              "similar_profiles": sum(self.prof_KO_perm_results.similar) /
                                               len(self.prof_KO_perm_results),
                                               "unsimilar": sum(self.prof_KO_perm_results.unsimilar) /
                                               len(self.prof_KO_perm_results)})

    def e_val_calc(self):
        """Return Ortho_Stats.e_value (int) which is an expected number of
        interactions with positive DMF and similar gene profiles by chance.
        """
        self.e_value = (self.DMF_positive_num * self.sim_prof_num) / self.tot_inter_num
