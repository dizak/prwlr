# -*- coding: utf-8 -*-


import re
import pathos.multiprocessing as ptmp
import numpy as np
import pandas as pd
from apis import KEGG_API as _KEGG_API
from errors import ParserError
from profiles import Profile as _Profile
from utils import remove_from_list


class KEGG:
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
                entry_dict["entry"] = entry[0].replace("ENTRY", "").replace("KO", "").strip()
            name = re.findall("NAME.+", i)
            if len(name) > 0:
                entry_dict["name"] = name[0].replace("NAME", "").strip()
            definition = re.findall("DEFINITION.+", i)
            if len(definition):
                entry_dict["definition"] = definition[0].replace("DEFINITION", "").strip()
            reference = re.findall("REFERENCE.+", i)
            if len(reference) > 0:
                entry_dict["reference"] = reference[0].replace("REFERENCE", "").strip()
            authors = re.findall("AUTHORS.+", i)
            if len(authors) > 0:
                entry_dict["authors"] = authors[0].replace("AUTHORS", "").strip()
            title = re.findall("TITLE.+", i)
            if len(title) > 0:
                entry_dict["title"] = title[0].replace("TITLE", "").strip()
            journal = re.findall("JOURNAL.+", i)
            if len(journal) > 0:
                entry_dict["journal"] = journal[0].replace("JOURNAL", "").strip()
            sequence = re.findall("SEQUENCE.+", i)
            if len(sequence) > 0:
                entry_dict["sequence"] = sequence[0].replace("SEQUENCE", "").replace("[", "").replace("]", "").strip()
            genes_blk_comp = re.compile("GENES.+\n^\s+\w{3}:\s.+^\w", re.DOTALL | re.MULTILINE)
            genes_blk_list = genes_blk_comp.findall(i)
            re.purge()
            if len(genes_blk_list) > 0:
                genes_blk_str = genes_blk_list[0]
                orgs_n_genes = re.findall("\w{3}:.+", genes_blk_str)
                orgs = []
                genes = []
                for i in orgs_n_genes:
                    try:
                        orgs.append(i.split(": ")[0])
                        genes.append(i.split(": ")[1])
                    except:
                        orgs.append(i)
                entry_dict["genes"] = genes
                entry_dict["orgs"] = orgs
            return entry_dict
        listed = ptmp.ProcessingPool().map(f, entries_list)
        df = pd.DataFrame(listed)
        if cleanup is True:
            df = df.drop_duplicates(subset=["entry"],
                                    keep="first")
            df.index = range(len(df))
            df.dropna(how="all",
                      inplace=True)
            df.dropna(subset=["entry",
                              "orgs"],
                      inplace=True)
        if remove_from_orgs is not None:
            for i in remove_from_orgs:
                df["orgs"] = df["orgs"].apply(lambda x: remove_from_list(i, x))
        self.database = df

    def parse_organism_info(self,
                            organism,
                            reference_species,
                            IDs,
                            X_ref,
                            strip_kegg_id_prefix):
        self._api.get_organisms_ids(IDs, skip_dwnld=True)
        self.IDs_table = self._api.organisms_ids_df
        self.reference_species = [self._api.org_name_2_kegg_id(i) for i in reference_species
                                  if i not in self._api.query_ids_not_found]
        self.reference_species = [i.upper() for i in self.reference_species if i is not None]
        self._api.get_org_db_X_ref(organism=organism,
                                   target_db=self.database_type,
                                   out_file_name=X_ref,
                                   skip_dwnld=True,
                                   strip_kegg_id_prefix=strip_kegg_id_prefix)
        self.X_reference = self._api.org_db_X_ref_df

    def profilize(self,
                  name="profiles"):
        """
        Append the database with phylogenetic profiles.
        """
        self.database[name] = self.database["orgs"].apply(lambda x:
                                                          _Profile(x, self.reference_species).to_string())


class Orthology(KEGG):
    """
    Restructures and appends data from KO database.

    Parameters
    -------
    dataframe: pandas.DataFrame
        listed converted to DataFrame.
    """

    def __init__(self):
        KEGG.__init__(self)
        self.query_species = None
        self.dataframe = None

    def profilize(self,
                  species_ids,
                  remove_empty=True,
                  upperize_ids=True,
                  profile_list=False,
                  KO_list_2_df=True,
                  profiles_df=True,
                  remove_species_white_spaces=True,
                  deduplicate=True):
        """Return KEGG.listed (list of dict) or KEGG.dataframe (pandas.DataFrame)
        appended with profiles (list of str or str).

        Args:
            species_ids (list of str): KEGG's IDs (3-letters) of reference
            species upon which are built.
            remove_empty (bool): remove inplace None types from the species_ids
            (list) <True> (default)
            upperize_ids (bool): make the items from the species_ids (list)
            upper-case as it is in the KEGG.listed orgs key when <True>
            (default)
            profile_list (bool): return each profile as the list of separate
            "+" or "-" when <True> or as one str when <False> (default)
            KO_list_2_df (bool): convert KEGG.listed to pandas.DataFrame.
            Rows NOT containing profiles are removed and resulting
            pandas.DataFrame is reindexed as continuous int sequence!
            profiles_df (bool): append KEGG.dataframe with sign-per-column
            profiles list
        """
        if remove_empty is True:
            species_ids = [i for i in species_ids if i is not None]
        if upperize_ids is True:
            species_ids = [i.upper() for i in species_ids]
        for i in self.listed:
            if "orgs" in i.keys():
                profile = ["+" if ii in i["orgs"] else "-" for ii in species_ids]
                if profile_list is False:
                    profile = "".join(profile)
                else:
                    pass
                i["profile"] = profile
            else:
                pass
        if KO_list_2_df is True:
            self.dataframe = pd.DataFrame(self.listed)
            self.dataframe = self.dataframe[-self.dataframe["profile"].isnull()]
            self.dataframe.index = range(len(self.dataframe))
            if deduplicate is True:
                self.dataframe = self.dataframe.drop_duplicates(subset=["entry"],
                                                                keep="first")
                self.dataframe.index = range(len(self.dataframe))
            else:
                pass
            if profiles_df is True:
                if remove_species_white_spaces is True:
                    profs_df = pd.DataFrame(self.dataframe.profile.map(lambda x:
                                                                       [i for i in x])
                                                          .tolist(),
                                            columns=[i.replace(" ", "_") for i in self.query_species])
                else:
                    profs_df = pd.DataFrame(self.dataframe.profile.map(lambda x:
                                                                       [i for i in x])
                                                          .tolist(),
                                            columns=self.query_species)
                self.dataframe = pd.concat([self.dataframe, profs_df], axis=1)
                self.dataframe.index = range(len(self.dataframe))
            else:
                pass
        else:

            pass


class SGA1:
    """
    Port from interactions.Ortho_Interactions. Meant to work just with SGA v1.

    Notes
    -------

    """
    def __init__(self):
        self.names = {'Array_ORF': 'ORF_A',
                      'Array_gene_name': 'GENE_A',
                      'Array_SMF': 'SMF_A',
                      'Array_SMF_standard_deviation': 'SMF_SD_A',
                      'Query_ORF': 'ORF_Q',
                      'Query_gene_name': 'GENE_Q',
                      'Query_SMF': 'SMF_Q',
                      'Query_SMF_standard_deviation': 'SMF_SD_Q',
                      'DMF': 'DMF',
                      'DMF_standard_deviation': 'DMF_SD',
                      'Genetic_interaction_score': 'GIS',
                      'Standard_deviation': 'GIS_SD',
                      'p-value': 'GIS_P'}

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


class SGA2:
    """
    Port from interactions.Ortho_Interactions. Meant to work just with SGA v1.

    Notes
    -------

    """
    def __init__(self):
        self.names = {"Query_Strain_ID": "STR_ID_Q",
                      "Query_allele_name": "GENE_Q",
                      "Array_Strain_ID": "STR_ID_A",
                      "Array_allele_name": "GENE_A",
                      "Arraytype/Temp": "TEMP",
                      "Genetic_interaction_score_(ε)": "GIS",
                      "P-value": "GIS_P",
                      "Query_single_mutant_fitness_(SMF)": "SMF_Q",
                      "Array_SMF": "SMF_A",
                      "Double_mutant_fitness": "DMF",
                      "Double_mutant_fitness_standard_deviation": "DMF_SD"}

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
        ORF_Q_col = self.sga["STR_ID_Q"].str.split("_", expand=True)[0]
        ORF_A_col = self.sga["STR_ID_A"].str.split("_", expand=True)[0]
        ORF_Q_col.name = "ORF_Q"
        ORF_A_col.name = "ORF_A"
        self.sga = pd.concat([ORF_Q_col, ORF_A_col, self.sga], axis=1)


class Bioprocesses:
    """
    Port from interactions.Ortho_Interactions. Meant to work with
    bioprocesses_annotations.costanzo2009.
    """

    def __init__(self):
        self.names = ["ORF",
                      "GENE",
                      "BIOPROC"]

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


class ProfInt:
    """
    Concatenation of SGA and profilized KO.
    """
    def __init__(self):
        self.names = {"authors": "AUTH",
                      "definition": "DEF",
                      "entry": "ENTRY",
                      "genes": "GENES",
                      "journal": "JOURN",
                      "name": "NAME",
                      "orgs": "ORGS",
                      "profile": "PROF",
                      "reference": "REF",
                      "sequence": "SEQ",
                      "title": "TITLE"}

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
                               left_on="ORF_Q",
                               right_on="ORF_id",
                               how="left")
        self.merged = pd.merge(self.merged,
                               ORF_KO,
                               left_on="ORF_A",
                               right_on="ORF_id",
                               how="left",
                               suffixes=("_Q", "_A"))
        self.merged.drop(["ORF_id_Q", "ORF_id_A"],
                         axis=1,
                         inplace=True)
        self.merged.dropna(inplace=True)
        self.merged = pd.merge(self.merged,
                               KO_df,
                               left_on="kegg_id_Q",
                               right_on="ENTRY",
                               how="left")
        self.merged = pd.merge(self.merged,
                               KO_df,
                               left_on="kegg_id_A",
                               right_on="ENTRY",
                               how="left",
                               suffixes=('_Q', '_A'))
        self.merged.drop(["kegg_id_Q", "kegg_id_A"],
                         axis=1,
                         inplace=True)
        self.merged.dropna(inplace=True)
        for i in self.merged.itertuples():
            prof_1 = np.array(list(getattr(i, "PROF_Q")))
            prof_2 = np.array(list(getattr(i, "PROF_A")))
            temp_score_list.append(simple_profiles_scorer(prof_1,
                                                          prof_2))
        temp_score_df = pd.DataFrame(temp_score_list,
                                     index=self.merged.index,
                                     columns=["PSS"])
        self.merged = pd.concat([sga,
                                temp_score_df],
                                axis=1)
