# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import pathos.multiprocessing as ptmp


class Ortho_Interactions:
    """Holds data about gene interactions array extracted from (csv) file.
    Merges these data with Genome.gene_profiles (list of tuples) and
    Genome.genes (list of dicts) selected values or Genome.KO_df. All col names
    issues e.g. rename, are kept in here. Dicts for col names change are stored
    as dicts in Ortho_Interactions attribs.

    Attribs:
        query_species (tuple of strs): passed from Genome. Set to <None>
        if using data from KEGG
        genes (list of dicts): passed from Genome. Set to <None> if using
        data from KEGG
        gene_profiles (list of tuples): passed from Genome. Set to <None>
        if using data from KEGG
        sga1_heads (dict of strs): translation from variable name to
        Ortho_Interactions.inter_df. Meant to avoid hard-coding and shorten
        line length
        ORF_KO_df (pandas.DataFrame): passed from KEGG_API. Consists of 2
        columns - ORF name and KO orthology group ID
        inter_df (pandas.DataFrame): holds data about interactions from parsed
        csv file. Can be appended with Ortho_Interactions.gen_based_appender or
        Ortho_Interactions.ko_based_appender
    """
    def __init__(self,
                 query_species,
                 genes,
                 gene_profiles,
                 KO_df,
                 org_ortho_db_X_ref_df):
        self.query_species = query_species
        self.genes = genes
        self.gene_profiles = gene_profiles
        self.sga1_heads = {'Array_ORF': 'ORF_A',
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
        self.sga2_heads = {"Query_Strain_ID": "STR_ID_Q",
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
        self.bio_proc_heads = {"Gene_name": "GENE",
                               "Process": "BIOPROC"}
        self.KO_heads = {"authors": "AUTH",
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
        self.KO_df = KO_df
        self.ORF_KO_df = org_ortho_db_X_ref_df
        self.inter_df = None
        self.bio_proc_df = None

    def parse_sgadata(self,
                      in_file_name,
                      sga_ver=1,
                      excel=False,
                      p_value=float(0.05),
                      DMF_type="neutral",
                      remove_white_spaces=True,
                      in_sep=","):
        """Return Ortho_Interactions.interact_df (pandas.DataFrame) from
        parsed <csv> file. The minimal filtration is based of a given GIS_P
        and presence of DMF value. Further filtration results in DMF
        higher/lower than both SMFs.

        Args:
            in_file_name (str): name of file to parse
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
        print "\nreading in interactions csv...".format()
        if excel is False:
            sga_df = pd.read_csv(in_file_name, sep=in_sep)
        else:
            sga_df = pd.read_excel(in_file_name)
        if remove_white_spaces is True:
            sga_df.columns = [i.replace(" ", "_") for i in sga_df.columns]
        if sga_ver == 1:
            sga_df.rename(columns=self.sga1_heads, inplace=True)
        elif sga_ver == 2:
            sga_df.rename(columns=self.sga2_heads, inplace=True)
            ORF_Q_col = sga_df["STR_ID_Q"].str.split("_", expand=True)[0]
            ORF_A_col = sga_df["STR_ID_A"].str.split("_", expand=True)[0]
            ORF_Q_col.name = "ORF_Q"
            ORF_A_col.name = "ORF_A"
            sga_df = pd.concat([ORF_Q_col, ORF_A_col, sga_df], axis=1)
        else:
            pass
        positive_DMF_bool = (sga_df["DMF"] > sga_df["SMF_Q"]) &\
                            (sga_df["DMF"] > sga_df["SMF_A"]) &\
                            (sga_df["GIS_P"] <= p_value)
        negative_DMF_bool = (sga_df["DMF"] < sga_df["SMF_Q"]) &\
                            (sga_df["DMF"] < sga_df["SMF_A"]) &\
                            (sga_df["GIS_P"] <= p_value)
        neutral_DMF_bool = (sga_df["DMF"].notnull()) &\
                           (sga_df["GIS_P"] <= p_value)
        print "\nselecting data...".format()
        if DMF_type == "positive":
            self.inter_df = sga_df[positive_DMF_bool]
        elif DMF_type == "negative":
            self.inter_df = sga_df[negative_DMF_bool]
        elif DMF_type == "neutral":
            self.inter_df = sga_df[neutral_DMF_bool]
        elif DMF_type == "raw":
            self.inter_df = sga_df
        else:
            pass

    def parse_bioprocesses(self,
                           in_file_name,
                           excel=False,
                           in_sep=","):
        """Return Ortho_Interactions.bio_proc_df (pandas.DataFrame) from parsed
        <csv> or <xls> file.

        Args:
            in_file_name (str): name of file to parse
            excel (bool): pandas.read_excel when <True>. pandas.read_csv when
            <False> (default).
            in_sep (str): separator for pandas.read_csv method
        """
        if excel is False:
            self.bio_proc_df = pd.read_csv(in_file_name, sep=in_sep)
        else:
            self.bio_proc_df = pd.read_excel(in_file_name)
        self.bio_proc_df.rename(columns=self.bio_proc_heads, inplace=True)

    def gen_based_appender(self,
                           bio_proc=True,
                           profiles_df=True):
        """Return Ortho_Interactions.inter_df appended by concatenated
        Genome.gene_profiles (list of tuples), Genome.gene_profiles similarity
        score (float), Genome.genes(list of dicts) gene descriptors. Optionally
        appends with Genome.gene_profiles array browsable by organism's name.

        Args:
            bio_proc (bool): appends with Ortho_Interactions.bio_proc_df array
            when <True> (default)
            profiles_df (bool): appends with Genome.gene_profiles array when
            <True> (default). Removes <None> rows
        """
        qa_attrib_temp_list = []
        prof_score_temp_list = []
        conc_qa_prof_temp_list = []
        q_gene_name_temp_list = []
        a_gene_name_temp_list = []
        q_gene_head_temp_list = []
        a_gene_head_temp_list = []
        bio_proc_temp_list = []
        print "\ncreating attributes list...".format()
        for i in self.inter_df.itertuples():
            qa_attrib_temp_list.append([getattr(i, "GENE_Q"),
                                        getattr(i, "GENE_A")])
        print "\nscoring profiles similarity...".format()
        prof_score_temp_list = ptmp.ProcessingPool().map(lambda x: df_qa_names_2_prof_score(x,
                                                                                            self.gene_profiles),
                                                         qa_attrib_temp_list)
        print "\nconcatenating profiles...".format()
        conc_qa_prof_temp_list = ptmp.ProcessingPool().map(lambda x: [gene_profile_finder_by_name(x[0],
                                                                                                  self.gene_profiles,
                                                                                                  conc=True),
                                                                      gene_profile_finder_by_name(x[1],
                                                                                                  self.gene_profiles,
                                                                                                  conc=True)],
                                                           qa_attrib_temp_list)
        print "\npreparing descriptors of query genes...".format()
        for i in self.inter_df.itertuples():
            q_gene_name_temp_list.append(getattr(i, "GENE_Q"))
        q_gene_head_temp_list = ptmp.ProcessingPool().map(lambda x: gene_finder_by_attrib("GN_gene_id",
                                                                                          x,
                                                                                          "description",
                                                                                          self.genes_inter),
                                                          q_gene_name_temp_list)
        print "\npreparing descriptors of array genes...".format()
        for i in self.inter_df.itertuples():
            a_gene_name_temp_list.append(getattr(i, "GENE_A"))
        a_gene_head_temp_list = ptmp.ProcessingPool().map(lambda x: gene_finder_by_attrib("GN_gene_id",
                                                                                          x,
                                                                                          "description",
                                                                                          self.genes_inter),
                                                          a_gene_name_temp_list)
        print "\ncreating temporary dataframes...".format()
        prof_score_temp_df = pd.DataFrame(prof_score_temp_list,
                                          index=self.inter_df.index,
                                          columns=["PSS"])
        conc_q_a_prof_temp_df = pd.DataFrame(conc_qa_prof_temp_list,
                                             index=self.inter_df.index,
                                             columns=["PROF_Q", "PROF_A"])
        q_gene_head_temp_df = pd.DataFrame(q_gene_head_temp_list,
                                           index=self.inter_df.index,
                                           columns=["DESC_Q"])
        a_gene_head_temp_df = pd.DataFrame(a_gene_head_temp_list,
                                           index=self.inter_df.index,
                                           columns=["DESC_A"])
        print "\nconcatenating dataframes...".format()
        self.inter_df = pd.concat([self.inter_df,
                                   prof_score_temp_df,
                                   conc_q_a_prof_temp_df,
                                   q_gene_head_temp_df,
                                   a_gene_head_temp_df],
                                  axis=1)
        if bio_proc is True:
            print "\nappending with bioprocesses info...".format()
            self.inter_df = pd.merge(self.inter_df,
                                     self.bio_proc_df,
                                     left_on="GENE_Q",
                                     right_on="GENE",
                                     how="left")
            self.inter_df = pd.merge(self.inter_df,
                                     self.bio_proc_df,
                                     left_on="GENE_A",
                                     right_on="GENE",
                                     how="left",
                                     suffixes=("_Q", "_A"))
            self.inter_df.drop(["GENE_Q", "GENE_A"],
                               axis=1,
                               inplace=True)
            for i in self.inter_df.itertuples():
                if getattr(i, "BIOPROC_Q") == getattr(i, "BIOPROC_A"):
                    if getattr(i, "BIOPROC_Q") == "unknown" or\
                       getattr(i, "BIOPROC_Q") == "unknown":
                        bio_proc_temp_list.append("unknown")
                    else:
                        bio_proc_temp_list.append("identical")
                else:
                    bio_proc_temp_list.append("different")
            bio_proc_temp_df = pd.DataFrame(bio_proc_temp_list,
                                            index=self.inter_df.index,
                                            columns=["BSS"])
            self.inter_df = pd.concat([self.inter_df,
                                       bio_proc_temp_df],
                                      axis=1)
        else:
            pass
        if profiles_df is True:
            print "\nappending with sign-per-column profiles...".format()
            cols_query_temp_list = ["GENE_Q"] + list(self.query_species)
            cols_array_temp_list = ["GENE_A"] + list(self.query_species)
            sep_prof_temp_df = pd.DataFrame(self.gene_profiles,
                                            columns=cols_query_temp_list)
            self.inter_df = pd.merge(self.inter_df,
                                     sep_prof_temp_df,
                                     on="GENE_Q")
            sep_prof_temp_df.columns = cols_array_temp_list
            self.inter_df = pd.merge(self.inter_df,
                                     sep_prof_temp_df,
                                     on="GENE_A",
                                     suffixes=("_Q", "_A"))
        else:
            pass

    def KO_based_appender(self):
        """Return Ortho_Interactions.inter_df appended by
        Ortho_Interactions.KO_df. Merge key: ORF
        """
        temp_score_list = []
        self.KO_df.rename(columns=self.KO_heads,
                          inplace=True)
        self.inter_df = pd.merge(self.inter_df,
                                 self.ORF_KO_df,
                                 left_on="ORF_Q",
                                 right_on="ORF_id",
                                 how="left")
        self.inter_df = pd.merge(self.inter_df,
                                 self.ORF_KO_df,
                                 left_on="ORF_A",
                                 right_on="ORF_id",
                                 how="left",
                                 suffixes=("_Q", "_A"))
        self.inter_df.drop(["ORF_id_Q", "ORF_id_A"],
                           axis=1,
                           inplace=True)
        self.inter_df.dropna(inplace=True)
        self.inter_df = pd.merge(self.inter_df,
                                 self.KO_df,
                                 left_on="kegg_id_Q",
                                 right_on="ENTRY",
                                 how="left")
        self.inter_df = pd.merge(self.inter_df,
                                 self.KO_df,
                                 left_on="kegg_id_A",
                                 right_on="ENTRY",
                                 how="left",
                                 suffixes=('_Q', '_A'))
        self.inter_df.drop(["kegg_id_Q", "kegg_id_A"],
                           axis=1,
                           inplace=True)
        self.inter_df.dropna(inplace=True)
        for i in self.inter_df.itertuples():
            prof_1 = np.array(list(getattr(i, "PROF_Q")))
            prof_2 = np.array(list(getattr(i, "PROF_A")))
            temp_score_list.append(simple_profiles_scorer(prof_1,
                                                          prof_2))
        temp_score_df = pd.DataFrame(temp_score_list,
                                     index=self.inter_df.index,
                                     columns=["PSS"])
        self.inter_df = pd.concat([self.inter_df,
                                   temp_score_df],
                                  axis=1)

    def bio_proc_appender(self):
        """Return Ortho_Interactions.inter_df appended by
        Ortho_Interactions.KO_df. Merge key: GENE
        """
        bio_proc_temp_list = []
        self.inter_df = pd.merge(self.inter_df,
                                 self.bio_proc_df,
                                 left_on="GENE_Q",
                                 right_on="GENE",
                                 how="left")
        self.inter_df.drop(["GENE"],
                           axis=1,
                           inplace=True)
        self.inter_df = pd.merge(self.inter_df,
                                 self.bio_proc_df,
                                 left_on="GENE_A",
                                 right_on="GENE",
                                 how="left",
                                 suffixes=('_Q', '_A'))
        self.inter_df.drop(["GENE"],
                           axis=1,
                           inplace=True)
        for i in self.inter_df.itertuples():
            if getattr(i, "BIOPROC_Q") == getattr(i, "BIOPROC_A"):
                if getattr(i, "BIOPROC_Q") == "unknown" or\
                   getattr(i, "BIOPROC_Q") == "unknown":
                    bio_proc_temp_list.append("unknown")
                else:
                    bio_proc_temp_list.append("identical")
            else:
                bio_proc_temp_list.append("different")
        bio_proc_temp_df = pd.DataFrame(bio_proc_temp_list,
                                        index=self.inter_df.index,
                                        columns=["BSS"])
        self.inter_df = pd.concat([self.inter_df,
                                  bio_proc_temp_df],
                                  axis=1)
