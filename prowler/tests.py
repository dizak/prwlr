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
        self.kegg_api = p.KEGG_API()
        self.cost_api = p.Costanzo_API()
        self.orgs_ids_out = pd.read_csv("./test_data/test_orgs_ids_out.csv",
                                        sep="\t",
                                        dtype="object")
        self.orgs_ids_in = "./test_data/test_orgs_ids_in.csv"

    def test_get_organisms_ids(self):
        """
        Test if apis.get_organisms_ids return correct pandas.DataFrame from
        input csv.
        """
        self.kegg_api.get_organisms_ids(self.orgs_ids_in,
                                        skip_dwnld=True)
        pd.testing.assert_frame_equal(self.kegg_api.organisms_ids_df,
                                      self.orgs_ids_out)


if __name__ == '__main__':
    unittest.main()
