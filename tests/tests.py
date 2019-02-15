# -*- coding: utf-8 -*-


import unittest
import requests as rq
from prwlr import *
import pandas as pd
import numpy as np
import pickle
import os

def isUp(url):
    """
    Returns <True> if status code of the website/server is 200.

    Parameters
    -------
    url: str
        URL to check.

    Returns
    -------
        bool
    """
    try:
        res = rq.get(url)
    except:
        return False
    return True if res.status_code == 200 else False

class UtilsTests(unittest.TestCase):
    """
    Tests for prwlr.utils.
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.ref_combinations_number = 3
        self.ref_string = "abcdefghijk"
        self.test_string = "abcde[]',fghijk"
        self.characters_to_remove = "[]',"
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

    def test_remove_char(self):
        """
        Test if utils.remove_char returns correct value.
        """
        self.assertEqual(self.ref_string,
                         utils.remove_char(self.test_string,
                                           self.characters_to_remove))


class ApisTests(unittest.TestCase,
                apis.Columns):
    """
    Tests for prwlr.apis.
    """
    @classmethod
    def setUpClass(cls):
        """
        Sets up class level attributes for the tests.
        """
        super(ApisTests, cls).setUpClass()
        cls.orgs_ids_out = pd.read_csv("test_data/ApisTests/test_orgs_ids_out.csv",
                                        sep='\t',
                                        index_col=[0])
        cls.orgs_ids_out = cls.orgs_ids_out.astype({k: v for k, v in cls.dtypes.items()
                                                    if k in cls.orgs_ids_out.columns})
        cls.ref_org_db_X_df = pd.read_csv("test_data/ApisTests/ref_orgs_db_X_ref.csv",
                                           sep="\t",
                                           names=["ORF_ID",
                                                  "KEGG_ID"],
                                           dtype="object")
        cls.ref_KOs_db_X_ref_df = pd.read_csv(
            "test_data/ApisTests/ref_KOs_db_X_ref.csv",
            sep="\t",
            names=[apis.Columns.KEGG_ID,
                   apis.Columns.ORG_GENE_ID],
            index_col=[0]
            ).groupby(
                by=[apis.Columns.KEGG_ID]
                )[apis.Columns.ORG_GENE_ID].apply(list).to_frame().reset_index()
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
        cls.kegg_api.get_KOs_db_X_ref(target_db="genes",
                                      filename="test_data/ApisTests/test_KOs_db_X_ref.csv",
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
                                      self.ref_org_db_X_df)

    def test_get_KOs_db_X_ref(self):
        """
        Test if apis.get_KOs_db_X_ref returns correct KEGG database
        cross-reference.
        """
        pd.testing.assert_frame_equal(self.ref_KOs_db_X_ref_df,
                                      self.kegg_api.KOs_db_X_ref_df)



# @unittest.skipUnless(
#     all(
#         isUp(address)
#         for address in (
#             apis.CostanzoAPI().home['v1'],
#             apis.CostanzoAPI().home['v2'],
#         )
#     ),
#     'No connection',
#     )
# class CostanzoAPITests(unittest.TestCase):
#     """
#     Test of prwlr.apis.CostanzoAPI.
#     """
#     def setUp(self):
#         """
#         Sets up class level attributes for the tests.
#         """
#         self.costanzo_api = apis.CostanzoAPI()
#         self.sga_versions = ["v1", "v2"]
#
#     def tearDown(self):
#         """
#         Distroys files downloaded or created during the tests.
#         """
#         for sga_version in self.sga_versions:
#             for i in list(self.costanzo_api.data[sga_version].values()):
#                 os.remove("test_data/{}".format(i).replace("data_files/", "").replace("%20", "_").replace(":", "-"))
#
#     def test_get_data(self):
#         """
#         Tests if apis.CostanzoAPI,get_data downloads data files successfully.
#         """
#         for sga_version in self.sga_versions:
#             for i in list(self.costanzo_api.data[sga_version].keys()):
#                 self.costanzo_api.get_data(i, "test_data", sga_version)


class DatabasesTests(unittest.TestCase):
    """
    Tests for prwlr.databases.
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
        self.ref_databases_KEGG_ID_name = {'HPY': 'Helicobacter pylori',
                                           'MJA': 'Methanocaldococcus jannaschii',
                                           'MGE': 'Mycoplasma genitalium',
                                           'SCE': 'Saccharomyces cerevisiae',
                                           'BSU': 'Bacillus subtilis',
                                           'SYN': 'Synechocystis sp',
                                           'ECO': 'Escherichia coli',
                                           'MPN': 'Mycoplasma pneumoniae',
                                           'MTH': 'Methanothermobacter thermautotrophicus',
                                           'HIN': 'Haemophilus influenzae'}
        self.ref_databases_KEGG_name_ID = {'Haemophilus influenzae': 'HIN',
                                           'Bacillus subtilis': 'BSU',
                                           'Mycoplasma pneumoniae': 'MPN',
                                           'Escherichia coli': 'ECO',
                                           'Methanothermobacter thermautotrophicus': 'MTH',
                                           'Methanocaldococcus jannaschii': 'MJA',
                                           'Mycoplasma genitalium': 'MGE',
                                           'Helicobacter pylori': 'HPY',
                                           'Saccharomyces cerevisiae': 'SCE',
                                           'Synechocystis sp': 'SYN'}

        self.test_kegg_db_filename = "test_data/DatabasesTests/test_kegg_db"
        self.database_type = "Orthology"
        self.organism_name = "Saccharomyces cerevisiae"
        self.IDs = "test_data/ApisTests/test_orgs_ids_in.csv"
        self.X_ref = "test_data/ApisTests/test_orgs_db_X_ref.csv"
        self.KOs = "test_data/ApisTests/test_KOs_db_X_ref.csv"
        self.out_file_name = "test_data/ApisTests/test_orgs_db_X_ref.csv"
        self.ref_organism_info = pd.read_csv(
            "test_data/DatabasesTests/ref_organism_info.csv",
            sep="\t",
            index_col=[0],
            )
        self.ref_kegg_db = pd.read_csv("test_data/DatabasesTests/ref_kegg_db.csv",
                                       sep='\t',
                                       index_col=[0])
        self.kegg = databases.KEGG(self.database_type)
        self.ref_kegg_db[self.kegg.GENES] = self.ref_kegg_db[self.kegg.GENES].apply(lambda x: [_.strip()
                                                                                               for _ in x.replace("[", "").
                                                                                                          replace("]", "").
                                                                                                          replace("'", "").
                                                                                                          split(",")])
        self.ref_kegg_db[self.kegg.ORGS] = self.ref_kegg_db[self.kegg.ORGS].apply(lambda x: [_.strip()
                                                                                             for _ in x.replace("[", "").
                                                                                                  replace("]", "").
                                                                                                  replace("'", "").
                                                                                                  split(",")])

    def test_parse_database(self):
        """
        Test if kegg database if properly parsed.
        """
        self.kegg.parse_database(self.test_kegg_db_filename)
        pd.testing.assert_frame_equal(self.ref_kegg_db, self.kegg.database)

    def test_parse_organism_info(self):
        """
        Test if organisms info is properly parsed if input files are supplied.
        """
        self.kegg.parse_organism_info(organism=self.organism_name,
                                      reference_species=self.query_species,
                                      IDs=self.IDs,
                                      X_ref=self.X_ref,
                                      KOs=self.KOs)
        self.assertEqual(self.kegg.name_ID, self.ref_databases_KEGG_name_ID)
        self.assertEqual(self.kegg.ID_name, self.ref_databases_KEGG_ID_name)
        self.kegg.organism_info[self.kegg.PROF] = self.kegg.organism_info[self.kegg.PROF].apply(
            lambda x: x.to_string()
        )
        pd.testing.assert_frame_equal(self.kegg.organism_info, self.ref_organism_info)


class SGA1Tests(unittest.TestCase):
    """
    Tests of prwlr.SGA1
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.sga1 = databases.SGA1()
        self.ref_sga = pd.read_csv("test_data/SGA1Tests/ref_sga_v1_1000r.csv",
                                   sep='\t',
                                   index_col=[0])
        self.ref_sga = self.ref_sga.astype({k: v for k, v in self.sga1.dtypes.items()
                                            if k in self.ref_sga.columns})
        self.test_sga_filename = "test_data/SGA1Tests/test_sga_v1_1000r.csv"

    def test_parse(self):
        """
        Test if SGA_v1 input file is properly parsed.
        """
        self.sga1.parse(self.test_sga_filename)
        pd.testing.assert_frame_equal(self.ref_sga, self.sga1.sga)


class SGA2Tests(unittest.TestCase):
    """
    Tests for prwlr.databases.SGA2
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.sga2 = databases.SGA2()
        self.ref_sga = pd.read_csv("test_data/SGA2Tests/ref_sga_v2_1000r.csv")
        self.test_sga_filename = "test_data/SGA2Tests/test_sga_v2_1000r.csv"
        self.ref_sga = self.ref_sga.astype({k: v for k, v in self.sga2.dtypes.items()
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
    Tests for prwlr.databases.SGA2
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.ref_anynwrk = pd.read_csv('test_data/AnyNetworkTests/ref_anynetwork.csv',
                                       sep='\t',
                                       index_col=[0])
        self.test_anynwrk_filename = "test_data/AnyNetworkTests/test_anynetwork.xls"
        self.ORF_query_col = "genotype"
        self.ORF_array_col = "target"
        self.sheet_name = "de novo SNPs"
        self.anynwrk = databases.AnyNetwork()

    def test_parse(self):
        """
        Test if any interaction network in form of xls input file is properly parsed.

        Notes
        -------
        <old AA/new AA> column is dropped as pandas.testing.assert_frame_equal
        recognizes <0> and <NaN> values as different in compared dataframes for
        unknown reason.
        """
        self.anynwrk.parse(self.test_anynwrk_filename,
                           excel=True,
                           sheet_name=self.sheet_name,
                           ORF_query_col=self.ORF_query_col,
                           ORF_array_col=self.ORF_array_col)
        pd.testing.assert_frame_equal(self.ref_anynwrk.drop(columns=['old AA/new AA']),
                                      self.anynwrk.sga.drop(columns=['old AA/new AA']),
                                      check_column_type=False)


class BioprocessesTests(unittest.TestCase):
    """
    Tests for prwlr.databases.Bioprocesses
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.ref_bioprocesses = pd.read_csv("test_data/BioprocessesTests/ref_bioproc_100r.csv",
                                            sep='\t',
                                            index_col=[0])
        self.test_bioprocesses = pd.read_excel("test_data/BioprocessesTests/test_bioproc_100r.xls")

    def test_parse(self):
        """
        Test if bioprocesses file is properly parsed.
        """
        pd.testing.assert_frame_equal(self.ref_bioprocesses,
                                      self.test_bioprocesses)


class ProfileTests(unittest.TestCase):
    """
    Test for prwlr.profiles.Profile.
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.ref_query = list("acdfhiklostuz")
        self.ref_queries = [self.ref_query[:10],
                            list('!@hjlnrtwy'),
                            list('acdfhi@#$%'),
                            list("qoadzv!@#$")]
        self.ref_reference = list("bcefghijklmnprstuwxy")
        self.ref_reference_2 = list("ybcefghijklmnprstuwx")
        self.ref_bound = (("a", False),
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
                          ("z", False))
        self.ref_present = ("c", "f", "h", "i", "k", "l", "s", "t", "u")
        self.ref_absent = ("a", "d", "o", "z")
        self.alt_pos_sing = "$"
        self.alt_neg_sing = "#"
        self.ref_profile = "-+-+++++-+++-"
        self.ref_alt_profile = '#$#$$$$$#$$$#'
        self.methods = ["pairwise",
                        "jaccard",
                        "dice",
                        "hamming",
                        "kulsinski",
                        "rogerstanimoto",
                        "russellrao",
                        "sokalmichener"]
        self.ref_pss = [[10, 7, 5, 3],
                        [0.0, 0.3333333333333333, 0.625, 1.0],
                        [0.0, 0.2, 0.45454545454545453, 1.0],
                        [0.0, 0.3, 0.5, 0.7],
                        [0.3, 0.5384615384615384, 0.8, 1.0],
                        [0.0, 0.46153846153846156, 0.6666666666666666, 0.8235294117647058],
                        [0.3, 0.4, 0.7, 1.0],
                        [0.0, 0.46153846153846156, 0.6666666666666666, 0.8235294117647058]]
        self.ref_ignore_elements = ["a", "c", "f"]
        self.ref_pss_ignore = 10
        self.test_profile = profiles.Profile(reference=self.ref_reference,
                                             query=self.ref_query)
        self.test_profile_2 = profiles.Profile(reference=self.ref_reference_2,
                                               query=self.ref_query)
        self.test_pss_methods_profile = profiles.Profile(reference=self.ref_reference,
                                                         query=self.ref_query[:10])

    def test___eq__(self):
        """
        Test if two profiles instances are equal when using Python comparison.
        """
        self.assertEqual(self.test_profile, self.test_profile_2)

    def test__convert(self):
        """
        Test if profile is properly converted.
        """
        print(self.test_profile)
        print(self.test_profile._convert(positive_sign=self.alt_pos_sing,
                                         negative_sign=self.alt_neg_sing))
        self.assertEqual(self.test_profile._convert(positive_sign=self.alt_pos_sing,
                                                    negative_sign=self.alt_neg_sing),
                         tuple(self.ref_alt_profile))

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
        Test if Profiles Similarity Score (PSS) is properly calculated with
        jaccard distance measure.
        """
        for method, pss_grp in zip(self.methods, self.ref_pss):
            for query, pss in zip(self.ref_queries, pss_grp):
                self.assertEqual(self.test_pss_methods_profile.calculate_pss(profiles.Profile(reference=self.ref_reference,
                                                                                              query=query),
                                                                             method=method),
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
    Tests of prwlr.stats top-level functions.
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.ref_nwrk = pd.read_csv("test_data/StatsTests/ref_nwrk.csv", sep="\t")
        self.permutations_number = 10
        self.desired_pss = 14
        self.ref_PSS_sum = int(pd.DataFrame(self.ref_nwrk.groupby(by=[stats.Columns.PSS]).size()).sum())

    def test_binomial_pss_test(self):
        """
        Test if binomial test returns values within correct range.
        """
        self.assertGreater(stats.binomial_pss_test(desired_pss=self.desired_pss,
                                                   selected=self.ref_nwrk,
                                                   total=self.ref_nwrk)["average"],
                           11)
        self.assertLess(stats.binomial_pss_test(desired_pss=self.desired_pss,
                                                selected=self.ref_nwrk,
                                                total=self.ref_nwrk)["average"],
                        15)
        self.assertGreater(stats.binomial_pss_test(desired_pss=self.desired_pss,
                                                   selected=self.ref_nwrk,
                                                   total=self.ref_nwrk)["complete"],
                           550)
        self.assertLess(stats.binomial_pss_test(desired_pss=self.desired_pss,
                                                selected=self.ref_nwrk,
                                                total=self.ref_nwrk)["complete"],
                        650)


class CoreTests(unittest.TestCase):
    """
    Tests of prwlr.core functions.
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        import prwlr.core

        self.ref_reference_1 = list('bcefghijklmnprstuwxy')
        self.ref_query_1 = list('acdfhiklostuz')
        self.ref_reference_2 = list('acefghijklmnprstuwxy')
        self.ref_query_2 = list('abdfhiklostuz')
        self.ref_profiles_filename = 'test_data/CoreTests/ref_profiles.csv'
        self.ref_network_filename = 'test_data/CoreTests/ref_network.csv'
        self.test_saved_profiles_filename = 'test_data/CoreTests/test_save_profiles.csv'
        self.test_saved_network_filename = 'test_data/CoreTests/test_save_network.csv'

        self.ref_profile_1, self.ref_profile_2 = (
            prwlr.profiles.Profile(
                reference=self.ref_reference_1,
                query=self.ref_query_1,
            ),
            prwlr.profiles.Profile(
                reference=self.ref_reference_2,
                query=self.ref_query_2,
            ),
        )
        self.ref_profiles_srs = pd.Series(
            [self.ref_profile_1, self.ref_profile_2],
        )
        self.ref_network_df = pd.DataFrame(
            [{
                'ORF_Q': 'YAL001',
                'ORF_A': 'YAL002',
                'PROF_Q': self.ref_profile_1,
                'PROF_A': self.ref_profile_2,
                'PSS': self.ref_profile_1.calculate_pss(self.ref_profile_2),
                'ATTRIB_Q': 'attribute_query',
                'ATTRIB_A': 'attribute_array',
            }]
        )

    def tearDown(self):
        """
        Removes files created during the tests.
        """
        if os.path.exists(self.test_saved_profiles_filename):
            os.remove(self.test_saved_profiles_filename)
        if os.path.exists(self.test_saved_network_filename):
            os.remove(self.test_saved_network_filename)

    def test_read_profiles(self):
        """
        Tests if prwlr.core.read_profiles returns pandas.Series with correct
        prwlr.profiles.Profile objects.
        """
        import prwlr.core

        pd.testing.assert_series_equal(
            self.ref_profiles_srs,
            prwlr.core.read_profiles(
                self.ref_profiles_filename,
                index_col=[0],
            ),
        )

    def test_read_network(self):
        """
        Tests if prwlr.core.read_network returns pandas.DataFrame with correct
        prwlr.profiles.Profile objects.
        """
        import prwlr.core

        pd.testing.assert_frame_equal(
            self.ref_network_df,
            prwlr.core.read_network(
                self.ref_network_filename,
                index_col=[0],
            ),
            check_like=True,
        )

    def test_save_network(self):
        """
        Tests if prwlr.core.save_network saves network suitable for
        prwlr.core.read_network.
        """
        import prwlr.core

        prwlr.core.save_network(
            self.ref_network_df,
            self.test_saved_network_filename,
        )
        pd.testing.assert_frame_equal(
            self.ref_network_df,
            prwlr.core.read_network(
                self.test_saved_network_filename,
                index_col=[0],
                ),
            check_like=True,
        )

    def test_save_profiles(self):
        """
        Tests if prwlr.core.save_profiles writes CSV file with correct values.
        """
        import prwlr.core

        prwlr.core.save_profiles(
            self.ref_profiles_srs,
            self.test_saved_profiles_filename,
        )
        pd.testing.assert_series_equal(
            self.ref_profiles_srs,
            prwlr.core.read_profiles(
                self.test_saved_profiles_filename,
                index_col=[0],
            ),
        )


if __name__ == '__main__':
    unittest.main()
