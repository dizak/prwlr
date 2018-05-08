# -*- coding: utf-8 -*-


import unittest
from prowler import *
import pandas as pd
import numpy as np
import pickle
import os


class UtilsTests(unittest.TestCase):
    """
    Tests for prowler.utils.
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.ref_combinations_number = 3
        self.test_subset_size = 2
        self.test_set_size = 3
        self.test_reference_iterable = list("abcdefghijk")
        self.test_query_iterable_1 = list("dfij")
        self.test_query_iterable_2 = list("dfijz")

    def test_isiniterable(self):
        """
        Test if utils.isiniterable returns <True> or <False> properly depending
        on presence of elements in both iterables.
        """
        self.assertTrue(utils.isiniterable(self.test_query_iterable_1,
                                           self.test_reference_iterable))
        self.assertTrue(utils.isiniterable(self.test_query_iterable_2,
                                           self.test_reference_iterable))
        self.assertFalse(utils.isiniterable(self.test_query_iterable_2,
                                            self.test_reference_iterable,
                                            all_present=True))

    def test_remove_from_list(self):
        """
        Test if utils.remove_from_list returns list without unwanted element.
        """
        self.assertEqual(self.test_query_iterable_1,
                         utils.remove_from_list("z",
                                                self.test_query_iterable_2))

    def test_all_possible_combinations_counter(self):
        """
        Test if utils.all_possible_combinations_counter returns correct value.
        """
        self.assertEqual(self.ref_combinations_number,
                         utils.all_possible_combinations_counter(self.test_subset_size,
                                                                 self.test_set_size))


class ApisTests(unittest.TestCase):
    """
    Tests for prowler.apis.
    """
    @classmethod
    def setUpClass(cls):
        """
        Sets up class level attributes for the tests.
        """
        super(ApisTests, cls).setUpClass()
        cls.orgs_ids_out = pd.read_pickle("test_data/ApisTests/test_orgs_ids_out.pickle")
        cls.org_db_X_ref_out = pd.read_csv("test_data/ApisTests/test_orgs_db_X_ref.csv",
                                           sep="\t",
                                           names=["ORF_ID",
                                                  "KEGG_ID"],
                                           dtype="object")
        cls.orgs_names = ["Haemophilus influenzae",
                          "Mycoplasma genitalium",
                          "Methanocaldococcus jannaschii",
                          "Synechocystis sp",
                          "Saccharomyces cerevisiae",
                          "Mycoplasma pneumoniae",
                          "Escherichia coli",
                          "Helicobacter pylori",
                          "Methanothermobacter thermautotrophicus",
                          "Bacillus subtilis",
                          "Notus existans"]
        cls.orgs_ids = ["hin",
                        "mge",
                        "mja",
                        "syn",
                        "sce",
                        "mpn",
                        "eco",
                        "hpy",
                        "mth",
                        "bsu"]
        cls.kegg_api = apis.KEGG_API()
        cls.cost_api = apis.CostanzoAPI()
        cls.kegg_api.get_organisms_ids("test_data/ApisTests/test_orgs_ids_in.csv",
                                       skip_dwnld=True)
        cls.kegg_api.get_org_db_X_ref(organism="Saccharomyces cerevisiae",
                                      target_db="orthology",
                                      out_file_name="test_data/ApisTests/test_orgs_db_X_ref.csv",
                                      skip_dwnld=True)

    def test_get_organisms_ids(self):
        """
        Test if apis.get_organisms_ids returns correct pandas.DataFrame from
        input csv.
        """
        pd.testing.assert_frame_equal(self.kegg_api.organisms_ids_df,
                                      self.orgs_ids_out)

    def test_org_name_2_kegg_id(self):
        """
        Test if apis.org_name_2_kegg_id returns correct organism ID for
        biological name.
        """
        for org_name, org_id in zip(self.orgs_names, self.orgs_ids):
            self.assertEqual(self.kegg_api.org_name_2_kegg_id(org_name), org_id)

    def test_get_org_db_X_ref(self):
        """
        Test if apis.get_org_db_X_ref returns correct KEGG database
        cross-reference.
        """
        pd.testing.assert_frame_equal(self.kegg_api.org_db_X_ref_df,
                                      self.org_db_X_ref_out)


class CostanzoAPITests(unittest.TestCase):
    """
    Test of prowler.apis.CostanzoAPI.
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.costanzo_api = apis.CostanzoAPI()

    def tearDown(self):
        """
        Distroys files downloaded or created during the tests.
        """
        for i in self.costanzo_api.data.values():
            os.remove("test_data/{}".format(i))

    def test_get_data(self):
        """
        Tests if apis.CostanzoAPI,get_data downloads data files successfully.
        """
        for i in self.costanzo_api.data.keys():
            self.costanzo_api.get_data(i, "test_data")


class DatabasesTests(unittest.TestCase):
    """
    Tests for prowler.databases.
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.query_species = ["Haemophilus influenzae",
                              "Mycoplasma genitalium",
                              "Methanocaldococcus jannaschii",
                              "Synechocystis sp",
                              "Saccharomyces cerevisiae",
                              "Mycoplasma pneumoniae",
                              "Escherichia coli",
                              "Helicobacter pylori",
                              "Methanothermobacter thermautotrophicus",
                              "Bacillus subtilis",
                              "Notus existans"]
        self.query_ids = ["hin",
                          "mge",
                          "mja",
                          "syn",
                          "sce",
                          "mpn",
                          "eco",
                          "hpy",
                          "mth",
                          "bsu"]
        self.test_kegg_db_filename = "test_data/DatabasesTests/test_kegg_db"
        self.database_type = "Orthology"
        self.organism_name = "Saccharomyces cerevisiae"
        self.IDs = "test_data/ApisTests/test_orgs_ids_in.csv"
        self.X_ref = "test_data/ApisTests/test_orgs_db_X_ref.csv"
        self.out_file_name = "test_data/ApisTests/test_orgs_db_X_ref.csv"
        self.ref_kegg_db = pd.read_pickle("test_data/DatabasesTests/ref_kegg_db.pickle")
        with open("test_data/DatabasesTests/ref_databases_KEGG_name_ID.pickle", "rb") as fin:
            self.ref_databases_KEGG_name_ID = pickle.load(fin)
        with open("test_data/DatabasesTests/ref_databases_KEGG_ID_name.pickle", "rb") as fin:
            self.ref_databases_KEGG_ID_name = pickle.load(fin)
        self.kegg = databases.KEGG(self.database_type)

    def test_parse_database(self):
        """
        Test if KEGG database is properly parsed.
        """
        self.kegg.parse_database(self.test_kegg_db_filename)
        pd.testing.assert_frame_equal(self.kegg.database, self.ref_kegg_db)

    def test_parse_organism_info(self):
        """
        Test if organisms info is properly parsed.
        """
        self.kegg.parse_organism_info(organism=self.organism_name,
                                      reference_species=self.query_species,
                                      IDs=self.IDs,
                                      X_ref=self.X_ref)
        self.assertEqual(self.kegg.name_ID, self.ref_databases_KEGG_name_ID)
        self.assertEqual(self.kegg.ID_name, self.ref_databases_KEGG_ID_name)


class SGA1Tests(unittest.TestCase):
    """
    Tests of prowler.SGA1
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.sga1 = databases.SGA1()
        self.ref_sga_filename = "test_data/SGA1Tests/ref_sga_v1_1000r.pickle"
        self.ref_sga = pd.read_pickle(self.ref_sga_filename)
        self.test_sga_filename = "test_data/SGA1Tests/test_sga_v1_1000r.txt"

    def test_parse(self):
        """
        Test if SGA_v1 input file is properly parsed.
        """
        self.sga1.parse(self.test_sga_filename)
        pd.testing.assert_frame_equal(self.ref_sga, self.sga1.sga)


class SGA2Test(unittest.TestCase):
    """
    Tests for prowler.databases.SGA2
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.sga2 = databases.SGA2()
        self.ref_sga = pd.read_csv("test_data/SGA2Tests/ref_sga_v2_1000r.txt")
        self.test_sga_filename = "test_data/SGA2Tests/test_sga_v2_1000r.txt"
        self.ref_sga = self.ref_sga.astype({k: v for k, v in self.sga2.dtypes.iteritems()
                                            if k in self.ref_sga.columns})

    def test_parse(self):
        """
        Test if SGA_v2 input file is properly parsed.
        """
        self.sga2.parse(self.test_sga_filename)
        pd.testing.assert_frame_equal(self.ref_sga,
                                      self.sga2.sga)


class AnyNetworkTests(unittest.TestCase):
    """
    Tests for prowler.databases.SGA2
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.ref_anynwrk = pd.read_pickle("test_data/AnyNetworkTests/ref_anynetwork.pickle")
        self.test_anynwrk_filename = "test_data/AnyNetworkTests/test_anynetwork.xls"
        self.ORF_query_col = "genotype"
        self.ORF_array_col = "target"
        self.sheet_name = "de novo SNPs"
        self.anynwrk = databases.AnyNetwork()

    def test_parse(self):
        """
        Test if any interaction network in form of xls input file is properly parsed.
        """
        self.anynwrk.parse(self.test_anynwrk_filename,
                           excel=True,
                           sheet_name=self.sheet_name,
                           ORF_query_col=self.ORF_query_col,
                           ORF_array_col=self.ORF_array_col)
        pd.testing.assert_frame_equal(self.ref_anynwrk,
                                      self.anynwrk.sga,
                                      check_dtype=False,
                                      check_names=False)


class BioprocessesTests(unittest.TestCase):
    """
    Tests for prowler.databases.Bioprocesses
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.ref_bioprocesses_filename = "test_data/BioprocessesTests/ref_bioproc_100r.pickle"
        self.test_bioprocesses_filename = "test_data/BioprocessesTests/test_bioproc_100r.xls"
        self.ref_bioprocesses = pd.read_pickle(self.ref_bioprocesses_filename)
        self.test_bioprocesses = pd.read_excel(self.test_bioprocesses_filename)

    def test_parse(self):
        """
        Test if bioprocesses file is properly parsed.
        """
        pd.testing.assert_frame_equal(self.ref_bioprocesses,
                                      self.test_bioprocesses)


class ProfIntTests(unittest.TestCase):
    """
    Tests for prowler.databases.ProfInt
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.profint = databases.ProfInt()
        self.ref_merged = pd.read_pickle("test_data/ProfIntTests/ref_merged.pickle")
        self.ref_profilized_nwrk = pd.read_pickle("test_data/StatsTests/ref_nwrk.pickle").reset_index(drop=True)
        self.reference_species = self.ref_profilized_nwrk[self.profint.PROF_Q].iloc[0].query
        self.test_non_profilized_nwrk = self.ref_profilized_nwrk.drop([self.profint.PROF_Q,
                                                                       self.profint.PROF_A,
                                                                       self.profint.PSS],
                                                                      axis=1)
        self.test_kegg_db = pd.read_pickle("test_data/ProfIntTests/test_kegg_database.pickle")
        self.test_X_reference = pd.read_pickle("test_data/ProfIntTests/test_X_reference.pickle")
        self.test_sga = pd.read_pickle("test_data/ProfIntTests/test_sga.pickle")

    def test_merger(self):
        """
        Test if prowler.databases.SGA1.sga or prowler.databases.SGA2.sga is
        properly merged with prowler.apis.org_db_X_ref_df and
        prowler.databases.KEGG.database
        """
        self.profint.merger(self.test_kegg_db,
                            self.test_X_reference,
                            self.test_sga)
        pd.testing.assert_frame_equal(self.ref_merged, self.profint.merged)

    def test_profilize(self):
        """
        Tests if databases.ProfInt.merged is properly appended with
        profiles.Profile.
        """
        self.profint.merged = self.test_non_profilized_nwrk
        self.profint.profilize(self.reference_species)
        pd.testing.assert_series_equal(self.ref_profilized_nwrk[self.profint.PROF_Q].apply(lambda x: sorted(x.get_present())),
                                       self.profint.merged[self.profint.PROF_Q].apply(lambda x: sorted(x.get_present())))
        pd.testing.assert_series_equal(self.ref_profilized_nwrk[self.profint.PROF_Q].apply(lambda x: sorted(x.get_absent())),
                                       self.profint.merged[self.profint.PROF_Q].apply(lambda x: sorted(x.get_absent())))


class ProfileTests(unittest.TestCase):
    """
    Test for prowler.profiles.Profile.
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.ref_query = list("acdfhiklostuz")
        self.ref_queries = [self.ref_query,
                            "aaaaaaaaaaaaa",
                            "bbbbbbbbbbbbb"]
        self.ref_reference = list("bcefghijklmnprstuwxy")
        self.ref_bound = [("a", False),
                          ("c", True),
                          ("d", False),
                          ("f", True),
                          ("h", True),
                          ("i", True),
                          ("k", True),
                          ("l", True),
                          ("o", False),
                          ("s", True),
                          ("t", True),
                          ("u", True),
                          ("z", False)]
        self.ref_present = ["c", "f", "h", "i", "k", "l", "s", "t", "u"]
        self.ref_absent = ["a", "d", "o", "z"]
        self.alt_pos_sing = "$"
        self.alt_neg_sing = "#"
        self.ref_profile = "-+-+++++-+++-"
        self.ref_alt_profile = "#$#$$$$$#$$$#"
        self.ref_pss = [13, 4, 9]
        self.ref_ignore_elements = ["a", "c", "f"]
        self.ref_pss_ignore = 10
        self.test_profile = profiles.Profile(reference=self.ref_reference,
                                             query=self.ref_query)

    def test__convert(self):
        """
        Test if profile is properly converted.
        """
        self.assertEqual(self.test_profile._convert(positive_sign=self.alt_pos_sing,
                                                    negative_sign=self.alt_neg_sing),
                         list(self.ref_alt_profile))

    def test__bind(self):
        """
        Test if profile values binds properly to profile.query
        """
        self.assertEqual(self.test_profile._bind(), self.ref_bound)

    def test_isall(self):
        """
        Test if Profile.isall returns True of False properly.
        """
        self.assertTrue(self.test_profile.isall(self.ref_present))
        self.assertFalse(self.test_profile.isall(self.ref_absent))

    def test_isany(self):
        """
        Test if Profile.isany returns True of False properly.
        """
        self.assertTrue(self.test_profile.isany(self.ref_present[:1] +
                                                self.ref_absent))
        self.assertFalse(self.test_profile.isany(self.ref_absent))

    def test_to_string(self):
        """
        Test if profile is properly converted to string.
        """
        self.assertEqual(self.test_profile.to_string(),
                         self.ref_profile)

    def test_to_list(self):
        """
        Test if profile is properly converted to list.
        """
        self.assertEqual(self.test_profile.to_list(),
                         list(self.ref_profile))
        self.assertEqual(self.test_profile.to_list(positive_sign=self.alt_pos_sing,
                                                   negative_sign=self.alt_neg_sing),
                         list(self.ref_alt_profile))

    def test_to_tuple(self):
        """
        Test if profile is properly converted to list.
        """
        self.assertEqual(self.test_profile.to_tuple(),
                         tuple(self.ref_profile))
        self.assertEqual(self.test_profile.to_tuple(positive_sign=self.alt_pos_sing,
                                                    negative_sign=self.alt_neg_sing),
                         tuple(self.ref_alt_profile))

    def test_to_array(self):
        """
        Test if profile is properly converted to numpy.array.
        """
        np.testing.assert_array_equal(self.test_profile.to_array(),
                                      np.array(list(self.ref_profile)))

    def test_to_series(self):
        """
        Test if profile is properly converted to pandas.Series.
        """
        pd.testing.assert_series_equal(self.test_profile.to_series(),
                                       pd.Series(list(self.ref_profile)))

    def test_calculate_pss(self):
        """
        Test if Profiles Similarity Score (PSS) is properly calculated.
        """
        for query, pss in zip(self.ref_queries, self.ref_pss):
            self.assertEqual(self.test_profile.calculate_pss(profiles.Profile(reference=self.ref_reference,
                                                                              query=query)),
                             pss)

    def test_calculate_pss_ignore(self):
        """
        Test if Profiles Similarity Score (PSS) is properly calculated with
        ignore arg used.
        """
        self.assertEqual(self.test_profile.calculate_pss(profiles.Profile(reference=self.ref_reference,
                                                                          query=self.ref_query),
                                                         ignore=self.ref_ignore_elements),
                         self.ref_pss_ignore)

    def test_get_present(self):
        """
        Test if get_present returns list of present items in the profile.
        """
        self.assertEqual(self.test_profile.get_present(), self.ref_present)

    def test_get_absent(self):
        """
        Test if get_absent returns list of present items in the profile.
        """
        self.assertEqual(self.test_profile.get_absent(), self.ref_absent)


class StatsTests(unittest.TestCase):
    """
    Test for prowler.stats.
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.profiles_similarity_threshold = 14
        self.p_value = 0.05
        self.GIS_min = 0.04
        self.GIS_max = -0.04
        self.query_species_selector = None
        self.array_species_selector = None
        self.ref_nwrk = pd.read_pickle("test_data/StatsTests/ref_nwrk.pickle").reset_index(drop=True)
        self.statistics = stats.Stats(self.ref_nwrk, 14)
        self.ref_nwrk_str = pd.read_csv("test_data/StatsTests/ref_nwrk.csv")
        self.flat_plu = "+" * 16
        self.flat_min = "-" * 16
        self.ref_PSS_sum = int(pd.DataFrame(self.ref_nwrk.groupby(by=[self.statistics.PSS]).size()).sum())
        self.permutations_number = 10

    def test_flat_plu_q(self):
        """
        Test if flat_plu_q selector returns dataframe of same length as
        selection on str.
        """
        pd.testing.assert_series_equal(self.statistics.dataframe[self.statistics.flat_plu_q]
                                       [self.statistics.PROF_Q].apply(lambda x: x.to_string()),
                                       self.ref_nwrk_str[self.ref_nwrk_str[self.statistics.PROF_Q] ==
                                                         self.flat_plu][self.statistics.PROF_Q])

    def test_flat_plu_a(self):
        """
        Test if flat_plu_a selector returns dataframe of same length as
        selection on str.
        """
        pd.testing.assert_series_equal(self.statistics.dataframe[self.statistics.flat_plu_a]
                                       [self.statistics.PROF_A].apply(lambda x: x.to_string()),
                                       self.ref_nwrk_str[self.ref_nwrk_str[self.statistics.PROF_A] ==
                                                         self.flat_plu][self.statistics.PROF_A])

    def test_flat_min_q(self):
        """
        Test if flat_min_q selector returns dataframe of same length as
        selection on str.
        """
        pd.testing.assert_series_equal(self.statistics.dataframe[self.statistics.flat_min_q]
                                       [self.statistics.PROF_Q].apply(lambda x: x.to_string()),
                                       self.ref_nwrk_str[self.ref_nwrk_str[self.statistics.PROF_Q] ==
                                                         self.flat_min][self.statistics.PROF_Q])

    def test_flat_min_a(self):
        """
        Test if flat_min_a selector returns dataframe of same length as
        selection on str.
        """
        pd.testing.assert_series_equal(self.statistics.dataframe[self.statistics.flat_min_a]
                                       [self.statistics.PROF_A].apply(lambda x: x.to_string()),
                                       self.ref_nwrk_str[self.ref_nwrk_str[self.statistics.PROF_A] ==
                                                         self.flat_min][self.statistics.PROF_A])

    def test_no_flat_plu_q(self):
        """
        Test if no_flat_plu_q selector returns dataframe of same length as
        selection on str.
        """
        pd.testing.assert_series_equal(self.statistics.dataframe[self.statistics.no_flat_plu_q]
                                       [self.statistics.PROF_Q].apply(lambda x: x.to_string()),
                                       self.ref_nwrk_str[self.ref_nwrk_str[self.statistics.PROF_Q] !=
                                                         self.flat_plu][self.statistics.PROF_Q])

    def test_no_flat_plu_a(self):
        """
        Test if no_flat_plu_a selector returns dataframe of same length as
        selection on str.
        """
        pd.testing.assert_series_equal(self.statistics.dataframe[self.statistics.no_flat_plu_a]
                                       [self.statistics.PROF_A].apply(lambda x: x.to_string()),
                                       self.ref_nwrk_str[self.ref_nwrk_str[self.statistics.PROF_A] !=
                                                         self.flat_plu][self.statistics.PROF_A])

    def test_no_flat_min_q(self):
        """
        Test if no_flat_min_q selector returns dataframe of same length as
        selection on str.
        """
        pd.testing.assert_series_equal(self.statistics.dataframe[self.statistics.no_flat_min_q]
                                       [self.statistics.PROF_Q].apply(lambda x: x.to_string()),
                                       self.ref_nwrk_str[self.ref_nwrk_str[self.statistics.PROF_Q] !=
                                                         self.flat_min][self.statistics.PROF_Q])

    def test_no_flat_min_a(self):
        """
        Test if no_flat_min_a selector returns dataframe of same length as
        selection on str.
        """
        pd.testing.assert_series_equal(self.statistics.dataframe[self.statistics.no_flat_min_a]
                                       [self.statistics.PROF_A].apply(lambda x: x.to_string()),
                                       self.ref_nwrk_str[self.ref_nwrk_str[self.statistics.PROF_A] !=
                                                         self.flat_min][self.statistics.PROF_A])

    def test_permute_profiles(self):
        """
        Test if permute_profiles returns proper dataframes using
        multiprocessing.
        """
        self.permuted = self.statistics.permute_profiles(self.ref_nwrk,
                                                         self.permutations_number,
                                                         return_series=True)
        for i in range(self.permutations_number):
            self.assertEqual(self.ref_PSS_sum, int(self.permuted[i].sum()))

    def test_permute_profiles_multiprocessing(self):
        """
        Test if permute_profiles returns proper dataframes using
        multiprocessing.
        """
        self.permuted = self.statistics.permute_profiles(self.ref_nwrk,
                                                         self.permutations_number,
                                                         return_series=True,
                                                         multiprocessing=True,
                                                         mp_backend="pathos")
        for i in range(self.permutations_number):
            self.assertEqual(self.ref_PSS_sum, int(self.permuted[i].sum()))

    def test_binomial_pss_test(self):
        """
        Test if binomial test returns values within correct range.
        """
        self.assertGreater(self.statistics.binomial_pss_test(desired_pss=14,
                                                             selected=self.ref_nwrk,
                                                             total=self.ref_nwrk)["average"],
                           11)
        self.assertLess(self.statistics.binomial_pss_test(desired_pss=14,
                                                          selected=self.ref_nwrk,
                                                          total=self.ref_nwrk)["average"],
                        15)
        self.assertGreater(self.statistics.binomial_pss_test(desired_pss=14,
                                                             selected=self.ref_nwrk,
                                                             total=self.ref_nwrk)["complete"],
                           550)
        self.assertLess(self.statistics.binomial_pss_test(desired_pss=14,
                                                          selected=self.ref_nwrk,
                                                          total=self.ref_nwrk)["complete"],
                        650)


if __name__ == '__main__':
    unittest.main()
