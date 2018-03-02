# -*- coding: utf-8 -*-


import unittest
from prowler import *
import pandas as pd
import subprocess as sp
import json


class ApisTest(unittest.TestCase):
    """
    Tests for prowler.apis.
    """
    @classmethod
    def setUpClass(cls):
        """
        Sets up class level attributes for the tests.
        """
        super(ApisTest, cls).setUpClass()
        cls.orgs_ids_out = pd.read_csv("./test_data/test_orgs_ids_out.csv",
                                       sep="\t",
                                       dtype="object")
        cls.org_db_X_ref_out = pd.read_csv("./test_data/test_orgs_db_X_ref.csv",
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
                          "Bacillus subtilis"]
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
        cls.cost_api = apis.Costanzo_API()
        cls.kegg_api.get_organisms_ids("./test_data/test_orgs_ids_in.csv",
                                       skip_dwnld=True)
        cls.kegg_api.get_org_db_X_ref(organism="Saccharomyces cerevisiae",
                                      target_db="orthology",
                                      out_file_name="./test_data/test_orgs_db_X_ref.csv",
                                      skip_dwnld=True)
        '''cls.kegg_api.get_db_entries("./test_data/test_db")
        with open("./test_data/test_db") as fin:
            cls.test_db = fin.read()
        with open("./test_data/test_db_ref") as fin:
            cls.test_db_ref = fin.read()'''

    @classmethod
    # def tearDownClass(cls):
    #     """
    #     Destroys database downloaded during tests.
    #     """
    #     sp.call("rm ./test_data/test_db", shell=True)
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
            self.kegg_api.org_name_2_kegg_id(org_name, org_id)

    def test_get_org_db_X_ref(self):
        """
        Test if apis.get_org_db_X_ref returns correct KEGG database
        cross-reference.
        """
        pd.testing.assert_frame_equal(self.kegg_api.org_db_X_ref_df,
                                      self.org_db_X_ref_out)

    '''def test_get_db_entries(self):
        """
        Test if apis.get_db_entries returns correct KEGG database.
        """
        self.assertEqual(self.test_db, self.test_db_ref, "test_db and test_db_ref are not equal.")'''


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
                              "Bacillus subtilis"]
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
        with open("./test_data/test_orthology_profilized.json") as fin:
            self.orthology_listed_ref = json.load(fin)


class SGA2Test(unittest.TestCase):
    """
    Tests for prowler.databases.SGA2
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.sga2 = databases.SGA2()
        self.test_sga_filename = "./test_data/test_sga_v2_1000r.txt"
        self.ref_sga = pd.read_csv("./test_data/ref_sga_v2_1000r.txt")
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
        self.test_anynwrk_filename = "./test_data/test_anynetwork.xls"
        self.ORF_query_col = "genotype"
        self.ORF_array_col = "target"
        self.sheet_name = "de novo SNPs"
        self.anynwrk = databases.AnyNetwork()
        self.ref_anynwrk = pd.read_csv("./test_data/ref_anynetwork.csv")

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


if __name__ == '__main__':
    unittest.main()
