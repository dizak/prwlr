# -*- coding: utf-8 -*-


from __future__ import print_function
import re
import pathos.multiprocessing as ptmp
import numpy as np
import pandas as pd
import tempfile
from prwlr.apis import KEGG_API as _KEGG_API
from prwlr.apis import Columns as _ApisColumns
from prwlr.errors import *
from prwlr.profiles import Profile as _Profile
from prwlr.utils import *


class Columns(_ApisColumns):
    """
    Container for the columns names defined in this module.
    """
    # Query and Array suffixes for query and array discrimination.
    QUERY_SUF = "_Q"
    ARRAY_SUF = "_A"
    # apis column names suffixed.
    KEGG_ID_A = "{}{}".format(_ApisColumns.KEGG_ID, ARRAY_SUF)
    KEGG_ID_Q = "{}{}".format(_ApisColumns.KEGG_ID, QUERY_SUF)
    ORF_ID_A = "{}{}".format(_ApisColumns.ORF_ID, ARRAY_SUF)
    ORF_ID_Q = "{}{}".format(_ApisColumns.ORF_ID, QUERY_SUF)
    # SGAs column names
    ORF_A = "ORF{}".format(ARRAY_SUF)
    GENE_A = "GENE{}".format(ARRAY_SUF)
    SMF_A = "SMF{}".format(ARRAY_SUF)
    SMF_SD_A = "SMF_SD{}".format(ARRAY_SUF)
    ORF_Q = "ORF_Q".format(QUERY_SUF)
    GENE_Q = "GENE_Q".format(QUERY_SUF)
    SMF_Q = "SMF_Q".format(QUERY_SUF)
    SMF_SD_Q = "SMF_SD_Q".format(QUERY_SUF)
    DMF = "DMF"
    DMF_SD = "DMF_SD"
    GIS = "GIS"
    GIS_SD = "GIS_SD"
    GIS_P = "GIS_P"
    STR_ID_Q = "STR_ID_Q".format(QUERY_SUF)
    GENE_Q = "GENE_Q".format(QUERY_SUF)
    STR_ID_A = "STR_ID{}".format(ARRAY_SUF)
    TEMP = "TEMP"
    # Bioprocesses and permutation internal dataframe column names.
    ORF = "ORF"
    GENE = "GENE"
    BIOPROC = "BIOPROC"
    # KEGG and ProfInt and permutation internal dataframe column names
    AUTH = "AUTH"
    DEF = "DEF"
    ENTRY = "ENTRY"
    GENES = "GENES"
    JOURN = "JOURN"
    NAME = "NAME"
    ORGS = "ORGS"
    ORGS_A = "ORGS{}".format(ARRAY_SUF)
    ORGS_Q = "ORGS{}".format(QUERY_SUF)
    PROF = "PROF"
    REF = "REF"
    SEQ = "SEQ"
    TITLE = "TITLE"
    PSS = "PSS"
    PROF_Q = "PROF{}".format(QUERY_SUF)
    PROF_A = "PROF{}".format(ARRAY_SUF)
    dtypes = {GIS: "float32",
              GIS_SD: "float32",
              SMF_Q: "float32",
              SMF_A: "float32",
              DMF: "float32",
              DMF_SD: "float32",
              PSS: "uint8"}


class KEGG(Columns):
    """
    Parses data downloaded with prwlr.apis and restructures them.

    Parameters
    -------
    listed: list of dicts
        Data from parsed KEGG database.
    """
    def __init__(self,
                 database_type):
        self.database_type = database_type.lower()
        self._api = _KEGG_API()

    def parse_database(self,
                       filename,
                       cleanup=True,
                       remove_from_orgs=None):
        """Return KEGG.listed (list of dicts) which contains information from
        the file downloaded by KEGG_API.get_ortho_db_entries.

        Args:
            filename (str): file name to parse
        """
        with open(filename, "r") as fin:
            file_str = fin.read()
            entries_list = file_str.split("///")
            if len(entries_list) < 2:
                raise ParserError("No split sign. Check if <///> in file.")

        def f(i):
            entry_dict = {}
            entry = re.findall("ENTRY.+", i)
            if len(entry) > 0:
                entry_dict[self.ENTRY] = entry[0].replace("ENTRY", "").replace("KO", "").strip()
            name = re.findall("NAME.+", i)
            if len(name) > 0:
                entry_dict[self.NAME] = name[0].replace("NAME", "").strip()
            definition = re.findall("DEFINITION.+", i)
            if len(definition):
                entry_dict[self.DEF] = definition[0].replace("DEFINITION", "").strip()
            reference = re.findall("REFERENCE.+", i)
            if len(reference) > 0:
                entry_dict[self.REF] = reference[0].replace("REFERENCE", "").strip()
            authors = re.findall("AUTHORS.+", i)
            if len(authors) > 0:
                entry_dict[self.AUTH] = authors[0].replace("AUTHORS", "").strip()
            title = re.findall("TITLE.+", i)
            if len(title) > 0:
                entry_dict[self.TITLE] = title[0].replace("TITLE", "").strip()
            journal = re.findall("JOURNAL.+", i)
            if len(journal) > 0:
                entry_dict[self.JOURN] = journal[0].replace("JOURNAL", "").strip()
            sequence = re.findall("SEQUENCE.+", i)
            if len(sequence) > 0:
                entry_dict[self.SEQ] = sequence[0].replace("SEQUENCE", "").replace("[", "").replace("]", "").strip()
            genes_blk_comp = re.compile("GENES.+\n^\s+\w{3}:\s.+^\w", re.DOTALL | re.MULTILINE)
            genes_blk_list = genes_blk_comp.findall(i)
            re.purge()
            if len(genes_blk_list) > 0:
                genes_blk_str = genes_blk_list[0]
                orgs_n_genes = re.findall("\w{3}:.+", genes_blk_str)
                orgs = []
                genes = []
                for i in orgs_n_genes:
                    if ": " in i:
                        orgs.append(i.split(": ")[0])
                        genes.append(i.split(": ")[1])
                    else:
                        orgs.append(i)
                entry_dict[self.GENES] = genes
                entry_dict[self.ORGS] = orgs
            return entry_dict
        listed = ptmp.ProcessingPool().map(f, entries_list)
        df = pd.DataFrame(listed)
        if cleanup is True:
            df = df.drop_duplicates(subset=[self.ENTRY],
                                    keep="first")
            df.index = list(range(len(df)))
            df.dropna(how="all",
                      inplace=True)
            df.dropna(subset=[self.ENTRY,
                              self.ORGS],
                      inplace=True)
        if remove_from_orgs is not None:
            for i in remove_from_orgs:
                df[self.ORGS] = df[self.ORGS].apply(lambda x: remove_from_list(i, x))
        self.database = df

    def parse_organism_info(self,
                            organism,
                            reference_species,
                            restrict_to=None,
                            IDs=None,
                            X_ref=None,
                            KOs=None,
                            strip_prefix=True,
                            IDs_only=False,
                            drop_ORF_duplicates=True,
                            drop_KO_duplicates=True,
                            threads=6,
                            raise_exceptions=True):
        KOs_different = """{} of X_reference and KO_organisms are different.""".format(self.KEGG_ID)
        KOs_different_mltpl_threads_msg = """{} of X_reference and KO_organisms are different. This
        might be caused by the server access denial. Try
        deacreasing number of threads""".format(self.KEGG_ID)
        print("Getting the organisms' KEGG IDs...")
        if IDs:
            self._api.get_organisms_ids(IDs, skip_dwnld=True)
        else:
            IDs_tmp = tempfile.NamedTemporaryFile(delete=True)
            self._api.get_organisms_ids(IDs_tmp.name, skip_dwnld=False)
            IDs_tmp.close()
        self.reference_species = [self._api.org_name_2_kegg_id(i) for i in reference_species
                                  if i not in self._api.query_ids_not_found]
        self.reference_species = [i.upper() for i in self.reference_species if i is not None]
        self.name_ID = dict(list(zip([i for i in reference_species
                                 if i not in self._api.query_ids_not_found],
                                self.reference_species)))
        self.ID_name = dict(list(zip(self.reference_species, [i for i in reference_species
                                                         if i not in self._api.query_ids_not_found])))
        if IDs_only:
            return
        print("Getting the ORF-Orthology Group Cross Reference...")
        if X_ref:
            self._api.get_org_db_X_ref(organism=organism,
                                       target_db=self.database_type,
                                       out_file_name=X_ref,
                                       skip_dwnld=True,
                                       drop_ORF_duplicates=drop_ORF_duplicates,
                                       drop_KO_duplicates=drop_KO_duplicates,
                                       strip_prefix=True)
        else:
            X_ref_tmp = tempfile.NamedTemporaryFile(delete=True)
            self._api.get_org_db_X_ref(organism=organism,
                                       target_db=self.database_type,
                                       out_file_name=X_ref_tmp.name,
                                       drop_ORF_duplicates=drop_ORF_duplicates,
                                       drop_KO_duplicates=drop_KO_duplicates,
                                       skip_dwnld=False,
                                       strip_prefix=True)
            X_ref_tmp.close()
        if restrict_to is not None:
            self._api.org_db_X_ref_df = self._api.org_db_X_ref_df[
                self._api.org_db_X_ref_df[self._api.ORF_ID].isin(restrict_to)
            ]
        self.X_reference = self._api.org_db_X_ref_df
        print("Getting the Organisms List for Each of The Orthology Group...")
        if KOs:
            self._api.get_KOs_db_X_ref(filename=KOs,
                                       skip_dwnld=True)
        else:
            KOs_temp = tempfile.NamedTemporaryFile(delete=True)
            self._api.get_KOs_db_X_ref(filename=KOs_temp.name,
                                       skip_dwnld=False,
                                       squeeze=True,
                                       threads=threads)
            KOs_temp.close()
        try:
            pd.testing.assert_series_equal(
                self.X_reference[self.KEGG_ID].
                drop_duplicates().
                sort_values().
                reset_index(drop=True),
                self._api.KOs_db_X_ref_df[self.KEGG_ID].
                drop_duplicates().
                sort_values().
                reset_index(drop=True)
            )
            self.KO_organisms = self._api.KOs_db_X_ref_df
        except AssertionError:
            if threads > 1:
                if raise_exceptions:
                    raise ParserError(KOs_different_mltpl_threads_msg)
                else:
                    print(KOs_different_mltpl_threads_msg)
            else:
                if raise_exceptions:
                    raise ParserError(KOs_different)
                else:
                    print(KOs_different)
        self.organism_info = pd.merge(
            left=self.X_reference,
            right=self.KO_organisms,
            on=self.KEGG_ID,
        )
        self.organism_info[self.PROF] = self.organism_info[self.ORG_GENE_ID].apply(
            lambda x: _Profile(
                x,
                [i.lower() for i in self.name_ID.values()],
            )
        )
        self.organism_info.drop(
            columns=self.ORG_GENE_ID,
            inplace=True,
        )
        self.organism_info.drop_duplicates(inplace=True)



class SGA1(Columns):
    """
    Port from interactions.Ortho_Interactions. Meant to work just with SGA v1.

    Notes
    -------

    """
    def __init__(self):
        self.names = (('Query_ORF', self.ORF_Q),
                      ('Query_gene_name', self.GENE_Q),
                      ('Array_ORF', self.ORF_A),
                      ('Array_gene_name', self.GENE_A),
                      ('Genetic_interaction_score', self.GIS),
                      ('Standard_deviation', self.GIS_SD),
                      ('p-value', self.GIS_P),
                      ('Query_SMF', self.SMF_Q),
                      ('Query_SMF_standard_deviation', self.SMF_SD_Q),
                      ('Array_SMF', self.SMF_A),
                      ('Array_SMF_standard_deviation', self.SMF_SD_A),
                      ('DMF', self.DMF),
                      ('DMF_standard_deviation', self.DMF_SD))

    def parse(self,
              filename,
              remove_white_spaces=True,
              in_sep="\t",
              cleanup=True):
        """Return Ortho_Interactions.interact_df (pandas.DataFrame) from
        parsed <csv> file. The minimal filtration is based of a given GIS_P
        and presence of DMF value. Further filtration results in DMF
        higher/lower than both SMFs.

        Args:
            sga (str): name of file to parse
            sga_ver (int) = costanzo dataframe version
            excel (bool): pandas.read_excel when <True>. pandas.read_csv when
            <False> (default).
            p_value (float): maximum GIS_P for filtering
            DMF_type (str): positive -> DMF > both SMFs
                            negative -> DMF < both SMFs
                            neutral  -> DMF not <None> (default)
                            raw      -> no filter
            remove_white_spaces (bool): replaces whitespaces from col names
            with <_> when True (default)
            in_sep (str): separator for pandas.read_csv method
        """
        self.sga = pd.read_csv(filename,
                               sep=in_sep,
                               names=[k for k, v in self.names],
                               error_bad_lines=False,
                               warn_bad_lines=True)
        if remove_white_spaces is True:
            self.sga.columns = [i.replace(" ", "_") for i in self.sga.columns]
        self.sga.rename(columns=dict(self.names), inplace=True)
        self.sga = self.sga.astype({k: v for k, v in self.dtypes.items()
                                    if k in self.sga.columns})
        if cleanup:
            self.sga = self.sga.dropna().drop_duplicates().reset_index(drop=True)


class SGA2(Columns):
    """
    Port from interactions.Ortho_Interactions. Meant to work just with SGA v2.

    Notes
    -------

    """
    def __init__(self):
        self.names = {"Query_Strain_ID": self.STR_ID_Q,
                      "Query_allele_name": self.GENE_Q,
                      "Array_Strain_ID": self.STR_ID_A,
                      "Array_allele_name": self.GENE_A,
                      "Arraytype/Temp": self.TEMP,
                      "Genetic_interaction_score_(ε)": self.GIS,
                      "P-value": self.GIS_P,
                      "Query_single_mutant_fitness_(SMF)": self.SMF_Q,
                      "Array_SMF": self.SMF_A,
                      "Double_mutant_fitness": self.DMF,
                      "Double_mutant_fitness_standard_deviation": self.DMF_SD}

    def parse(self,
              filename,
              remove_white_spaces=True,
              in_sep="\t"):
        """Return Ortho_Interactions.interact_df (pandas.DataFrame) from
        parsed <csv> file. The minimal filtration is based of a given GIS_P
        and presence of DMF value. Further filtration results in DMF
        higher/lower than both SMFs.

        Args:
            filename (str): name of file to parse
            p_value (float): maximum GIS_P for filtering
            DMF_type (str): positive -> DMF > both SMFs
                            negative -> DMF < both SMFs
                            neutral  -> DMF not <None> (default)
                            raw      -> no filter
            remove_white_spaces (bool): replaces whitespaces from col names
            with <_> when True (default)
            in_sep (str): separator for pandas.read_csv method
        """
        self.sga = pd.read_csv(filename, sep=in_sep)
        if remove_white_spaces is True:
            self.sga.columns = [i.replace(" ", "_") for i in self.sga.columns]
        self.sga.rename(columns=self.names, inplace=True)
        ORF_Q_col = self.sga[self.STR_ID_Q].str.split("_", expand=True)[0]
        ORF_A_col = self.sga[self.STR_ID_A].str.split("_", expand=True)[0]
        ORF_Q_col.name = self.ORF_Q
        ORF_A_col.name = self.ORF_A
        self.sga = pd.concat([ORF_Q_col, ORF_A_col, self.sga], axis=1)
        self.sga = self.sga.astype({k: v for k, v in self.dtypes.items()
                                    if k in self.sga.columns})


class AnyNetwork(Columns):
    """
    Parses and holds data of any type of network.
    """
    def __init__(self):
        pass

    def parse(self,
              filename,
              sep="\t",
              excel=False,
              sheet_name=None,
              ORF_query_col=None,
              ORF_array_col=None,
              **kwargs):
        """
        Parses network csv file. Checks whether columns names in the csv file
        correspond with the databases.Columns.
        """
        if ORF_query_col is None or ORF_array_col is None:
            raise ParserError("No ORF_query or ORF_array column name.")
        if excel is True:
            self.sga = pd.read_excel(filename, sheet_name=sheet_name)
        else:
            self.sga = pd.read_csv(filename, sep=sep)
        self.sga.rename(columns={ORF_query_col: self.ORF_Q,
                                 ORF_array_col: self.ORF_A},
                        inplace=True)
        if len(kwargs) > 0:
            self.sga.rename(columns=kwargs,
                            inplace=True)


class Bioprocesses(Columns):
    """
    Port from interactions.Ortho_Interactions. Meant to work with
    bioprocesses_annotations.costanzo2009.
    """

    def __init__(self):
        self.names = [self.ORF,
                      self.GENE,
                      self.BIOPROC]

    def parse(self,
              filename):
        """Return Ortho_Interactions.bio_proc_df (pandas.DataFrame) from parsed
        <csv> or <xls> file.

        Parameters
        -------
        filename: str
            Name of file to parse.
        """
        self.bioprocesses = pd.read_excel(filename,
                                          names=self.names)
        self.bioprocesses = self.bioprocesses.astype({k: v for k, v in self.dtypes.items()
                                                     if k in self.bioprocesses.columns})
