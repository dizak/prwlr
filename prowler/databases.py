# -*- coding: utf-8 -*-


import re
import pathos.multiprocessing as ptmp
import pandas as pd


class KEGG:
    """
    Parses data downloaded with prowler.apis and restructures them.

    Parameters
    -------
    listed: list of dicts
        Data from parsed KEGG ortho-database.
    dataframe: pandas.DataFrame
        listed converted to DataFrame.
    """
    def __init__(self):
        self.listed = []
        self.dataframe = None

    def parse(self,
              filename):
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
        self.listed = ptmp.ProcessingPool().map(f, entries_list)

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
