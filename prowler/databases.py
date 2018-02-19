# -*- coding: utf-8 -*-


import re
import pathos.multiprocessing as ptmp
import numpy as np
import pandas as pd
from apis import KEGG_API as _KEGG_API
from apis import Columns as _ApisColumns
from errors import ParserError
from profiles import Profile as _Profile
from utils import *


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
    PROF_Q = "PROF_Q".format(QUERY_SUF)
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
    Parses data downloaded with prowler.apis and restructures them.

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
            df.index = range(len(df))
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
                            IDs,
                            X_ref,
                            strip_prefix=True):
        self._api.get_organisms_ids(IDs, skip_dwnld=True)
        self.reference_species = [self._api.org_name_2_kegg_id(i) for i in reference_species
                                  if i not in self._api.query_ids_not_found]
        self.reference_species = [i.upper() for i in self.reference_species if i is not None]
        self._api.get_org_db_X_ref(organism=organism,
                                   target_db=self.database_type,
                                   out_file_name=X_ref,
                                   skip_dwnld=True,
                                   strip_prefix=True)
        self.X_reference = self._api.org_db_X_ref_df

    def profilize(self):
        """
        Append the database with phylogenetic profiles.
        """
        self.database[self.PROF] = self.database[self.ORGS].apply(lambda x:
                                                                  _Profile(x, self.reference_species).to_string())


class SGA1(Columns):
    """
    Port from interactions.Ortho_Interactions. Meant to work just with SGA v1.

    Notes
    -------

    """
    def __init__(self):
        self.names = {'Array_ORF': self.ORF_A,
                      'Array_gene_name': self.GENE_A,
                      'Array_SMF': self.SMF_A,
                      'Array_SMF_standard_deviation': self.SMF_SD_A,
                      'Query_ORF': self.ORF_Q,
                      'Query_gene_name': self.GENE_Q,
                      'Query_SMF': self.SMF_Q,
                      'Query_SMF_standard_deviation': self.SMF_SD_Q,
                      'DMF': self.DMF,
                      'DMF_standard_deviation': self.DMF_SD,
                      'Genetic_interaction_score': self.GIS,
                      'Standard_deviation': self.GIS_SD,
                      'p-value': self.GIS_P}

    def parse(self,
              filename,
              remove_white_spaces=True,
              in_sep=","):
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
        self.sga = pd.read_csv(filename, sep=in_sep)
        if remove_white_spaces is True:
            self.sga.columns = [i.replace(" ", "_") for i in self.sga.columns]
        self.sga.rename(columns=self.names, inplace=True)
        self.sga = self.sga.astype({k: v for k, v in self.dtypes.iteritems()
                                    if k in self.sga.columns})


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
        self.sga = self.sga.astype({k: v for k, v in self.dtypes.iteritems()
                                    if k in self.sga.columns})


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
        self.bioprocesses = self.bioprocesses.astype({k: v for k, v in self.dtypes.iteritems()
                                                     if k in self.bioprocesses.columns})


class ProfInt(Columns):
    """
    Concatenation of SGA and profilized KO.
    """
    def __init__(self):
        self.names = {"authors": self.AUTH,
                      "definition": self.DEF,
                      "entry": self.ENTRY,
                      "genes": self.GENES,
                      "journal": self.JOURN,
                      "name": self.NAME,
                      "orgs": self.ORGS,
                      "profile": self.PROF,
                      "reference": self.REF,
                      "sequence": self.SEQ,
                      "title": self.TITLE}

    def merger(self,
               KO_df,
               ORF_KO,
               sga):
        """Return Ortho_Interactions.sga appended by
        Ortho_Interactions.KO_df. Merge key: ORF
        """
        temp_score_list = []
        KO_df.rename(columns=self.names,
                     inplace=True)
        self.merged = pd.merge(sga,
                               ORF_KO,
                               left_on=self.ORF_Q,
                               right_on=self.ORF_ID,
                               how="left")
        self.merged = pd.merge(self.merged,
                               ORF_KO,
                               left_on=self.ORF_A,
                               right_on=self.ORF_ID,
                               how="left",
                               suffixes=(self.QUERY_SUF, self.ARRAY_SUF))
        self.merged.drop([self.ORF_ID_Q,
                          self.ORF_ID_A],
                         axis=1,
                         inplace=True)
        self.merged.dropna(inplace=True)
        self.merged = pd.merge(self.merged,
                               KO_df,
                               left_on=self.KEGG_ID_Q,
                               right_on=self.ENTRY,
                               how="left")
        self.merged = pd.merge(self.merged,
                               KO_df,
                               left_on=self.KEGG_ID_A,
                               right_on=self.ENTRY,
                               how="left",
                               suffixes=(self.QUERY_SUF, self.ARRAY_SUF))
        self.merged.drop([self.KEGG_ID_Q,
                          self.KEGG_ID_A],
                         axis=1,
                         inplace=True)
        self.merged.dropna(inplace=True)
        # for i in self.merged.itertuples():
        #     prof_1 = np.array(list(getattr(i, "PROF_Q")))
        #     prof_2 = np.array(list(getattr(i, "PROF_A")))
        #     temp_score_list.append(simple_profiles_scorer(prof_1,
        #                                                   prof_2))
        # temp_score_df = pd.DataFrame(temp_score_list,
        #                              index=self.merged.index,
        #                              columns=["PSS"])
        # self.merged = pd.concat([sga,
        #                         temp_score_df],
        #                         axis=1)

    def profilize(self,
                  reference_species):
        """
        Append databases.merged with Profiles Similarity Score and/or string
        representation of the phylogenetic profiles.

        Parameters
        -------
        reference_species: list of str
            Species list compared to contained in the orthogroup. Basis for the
            profiles construction.
        """
        self.merged[self.PSS] = self.merged.apply(lambda x:
                                                  _Profile(x[self.ORGS_A],
                                                           reference_species).
                                                  calculate_pss(_Profile(x[self.ORGS_Q],
                                                                         reference_species)),
                                                  axis=1).astype(self.dtypes[self.PSS])
        self.merged[self.PROF_A] = self.merged[self.ORGS_A].apply(lambda x:
                                                                  _Profile(x, reference_species).to_string())
        self.merged[self.PROF_Q] = self.merged[self.ORGS_Q].apply(lambda x:
                                                                  _Profile(x, reference_species).to_string())
        self.merged = self.merged.astype({k: v for k, v in self.dtypes.iteritems()
                                          if k in self.merged.columns})
