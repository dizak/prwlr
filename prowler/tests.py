# -*- coding: utf-8 -*-


import unittest
import prowler as p
import pandas as pd


class ApisTest(unittest.TestCase):
    """
    Tests for prowler.apis.
    """
    def setUp(self):
        """
        Sets up class level attributes for the tests.
        """
        self.orgs_ids_out = pd.read_csv("./test_data/test_orgs_ids_out.csv",
                                        sep="\t",
                                        dtype="object")
        self.orgs_ids_in = "./test_data/test_orgs_ids_in.csv"
        self.orgs_names = ["Haemophilus influenzae",
                           "Mycoplasma genitalium",
                           "Methanocaldococcus jannaschii",
                           "Synechocystis sp",
                           "Saccharomyces cerevisiae",
                           "Mycoplasma pneumoniae",
                           "Escherichia coli",
                           "Helicobacter pylori",
                           "Methanothermobacter thermautotrophicus",
                           "Bacillus subtilis"]
        self.orgs_ids = ["hin",
                         "mge",
                         "mja",
                         "syn",
                         "sce",
                         "mpn",
                         "eco",
                         "hpy",
                         "mth",
                         "bsu"]
        self.kegg_api = p.KEGG_API()
        self.cost_api = p.Costanzo_API()
        self.kegg_api.get_organisms_ids(self.orgs_ids_in,
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
            self.kegg_api.org_name_2_kegg_id(org_name, org_id)


if __name__ == '__main__':
    unittest.main()
