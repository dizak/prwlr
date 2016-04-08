#! /usr/bin/env python

###############################################################################
#make profiles insensitive to id type
###############################################################################

__author__ = "Dariusz Izak"

import lxml.etree as et
import pandas as pd
import re
import math
import argparse
import glob
import sys

def prog_perc(in_item,
              in_iterbl):
    """Display progres of iterable as percent. Uses carriage return.

    Args:
        in_item: current iteration product
        in_iterbl: iterable
    """
    pos = in_iterbl.index(in_item)
    sys.stdout.write("{0}%\r".format(pos * 100 /\
                                     len(in_iterbl)))
    sys.stdout.flush()

def hash_prog(in_item,
              in_iterbl):
    tot_len = len(in_iterbl)
    ten_perc = tot_len / 10
    if i % ten_perc == 0:
        sys.stdout.write("#")
        sys.stdout.flush()

def all_possible_combinations_counter(in_int_set,
                                      in_int_subset):
    """Return a number (int) of all possible combinations of elements in size
    of a subset of a set.

    Args:
        in_int_set (int): size of the whole set
        in_int_subset (int): size of the subset
    """
    f = math.factorial
    return f(in_int_set) / f(in_int_subset) / f(in_int_set - in_int_subset)

def all_non_redundant_genes_lister(in_df,
                                   in_col_name_1,
                                   in_col_name_2):
    """Return a non-redundant set (set) of elements from taken from two pandas
    DataFrame columns merged together.

    Args:
        in_df (pandas.DataFrame): dataframe to take the columns from
        in_col_name_1 (str): name of the 1st column to take the elements from
        in_col_name_2 (str): name of the 2nd column to take the elements from
    """
    temp_list_1 = [getattr(i, "in_col_name_1") for i in in_df.itertuples()]
    temp_list_2 = [getattr(i, "in_col_name_2") for i in in_df.itertuples()]
    return set(temp_list_1 + temp_list_2)

def gene_finder_by_attrib(in_attr,
                          in_val,
                          out_attr,
                          in_genes,
                          exact = True):
    """Return an atrribute (dict value) of a gene from the Genome.genes found
    by another attribute of choice.

    Args:
        in_attr (str): attribute to search in
        in_val (str): desired value of the attribute to look for
        out_attr (str): attribute of which value is desired to be returned
        in_genes (list of dicts): Genome.genes to search against
        exact (bool): whole phrase when <True>. One word in phrase when <False>.
                      Default <True>

    Examples:
        >>> print gene_finder_by_attrib("prot_id", "P22139", "description",
                                        sc.genes)
        DNA-directed RNA polymerases I, II, and III subunit RPABC5
    """
    if exact == True:
        for i in in_genes:
            if i[in_attr] == in_val:
                return i[out_attr]
    else:
        for i in in_genes:
            if in_val in i[in_attr]:
                return i[out_attr]

def gene_profile_finder_by_name(in_gene_name,
                                in_all_gene_profiles,
                                conc = True):
    """Return a gene profile (tuple of str) starting from the second position
    within the tuple from Genome.gene_profiles found by its name.

    Args:
        in_gene_name (str): desired name to look for
        in_all_gene_profiles (list of tuples): Genome.gene_profiles to search
        against
        conc (bool): elements of profile merged when <True>. Unmodified when
        <False>

    Examples:
        >>> gene_profile_finder_by_name("SAL1", sc_gen.gene_profiles,
                                        conc = False)
        ('SAL1', '-', '-', '-', '+', '-', '+', '-', '+', '+', '-', '-', '+',
        '+', '+', '-', '-', '+', '+', '+')
        >>> gene_profile_finder_by_name("SAL1", sc_gen.gene_profiles,
                                        conc = False)
        '---+-+-++--+++--+++'
    """
    for i in set(in_all_gene_profiles):
        if in_gene_name == i[0]:
            if conc == True:
                return "".join(i[1:])
            else:
                return i

def gene_name_finder_by_profile(in_gene_profile,
                                in_all_gene_profiles):
    """Return a (list) of gene names (str) from Genome.gene_profiles found by
    (list) or (tuple) of a gene profile starting from the second position

    Args:
        in_gene_profile: desired profile to look for
        in_all_gene_profiles: Genome.gene_profiles to search against
    """
    in_gene_profile = tuple(in_gene_profile)
    return [i[0] for i in in_all_gene_profiles if i[1:] == in_gene_profile]

def simple_profiles_scorer(in_gene_profile_1,
                           in_gene_profile_2):
    """Return a score (int) which is a number of identicals between two gene
    profiles. Score equal zero means the profiles are mirrors.

    Args:
        in_gene_profile_1: the first profile to compare
        in_gene_profile_2: the second profile to compare
    """
    score = 0
    for i in range(1, len(in_gene_profile_1)):
        if in_gene_profile_1[i] == in_gene_profile_2[i]:
            score += 1
    return score

def df_qa_names_2_prof_score(in_genes_pair,
                             in_all_gene_profiles):
    """Return a score (int) ene profiles similarity using pair
    (list, tuple) of desired gene names or (None) if desired genes do not have
    profile.

    Args:
        in_genes_pair (list, tuple of str): genes to generate score for
        in_all_gene_profiles (list of tuples): Genome.gene_profiles to search
        against
    """
    gene_profile_1 = gene_profile_finder_by_name(in_genes_pair[0],
                                                 in_all_gene_profiles,
                                                 conc = False)
    gene_profile_2 = gene_profile_finder_by_name(in_genes_pair[1],
                                                 in_all_gene_profiles,
                                                 conc = False)
    if isinstance(gene_profile_1, type(None)) or isinstance(gene_profile_2,
                                                            type(None)) == True:
        pass
    else:
        return simple_profiles_scorer(gene_profile_1, gene_profile_2)

class Genome:
    """Holds data about the reference organism genome extracted from its
    proteome (multifasta) and its genes orthologs (OrthoXML, ortho-finder csv).
    Builds gene profiles.

    Atrribs:
        ref_name (str): reference organism's name
        query_species (list of strs) = organisms having orthologs with
        reference organism

        genes (list of dicts): key: values of the genes from parsed
        (multifasta) and their orthologs

        orthologous_groups_df (df): pandas DataFrame of ortho-finder csv

        orthologous_groups_dict (list of dicts): data extracted from
        orthologous_groups_df for convenience

        empty_genes (list of dicts): genes without orthologs

        ortho_genes (list of dicts): genes with orthologs

        gene_profiles (list of tuples): each profile contains gene name and +/-
        signs for presence/absence of ortholog in organism from
        Genome.query_species. Order is kept by holding data in tuples

    """
    def __init__(self):
        self.ref_name = None
        self.query_species = []
        self.genes = []
        self.orthologous_groups_df = None
        self.orthologous_groups_dict = []
        self.empty_genes = []
        self.ortho_genes = []
        self.gene_profiles = []

    def parse_fasta(self,
                    in_file_name):
        """Return Genome.genes (list of dicts) and Genome.ref_name.
        Genome.genes hold GN gene id, yeast gene id, protein id, heading,
        description and sequence from parsed (multifasta) for each gene.

        Args:
            in_file_name (str): name of (multifasta) file to parse
        """
        counter = 0
        tax_name_fasta_re_comp = re.compile("OS=\w+ \w+")
        prot_id_re_comp = re.compile("\|\S+?\|")
        yeast_gene_id_re_comp = re.compile("\w+?[_]\S+")
        GN_gene_id_re_comp = re.compile("(GN=\S+)")
        with open(in_file_name, "r") as fin:
            for i in fin:
                if i.startswith(">"):
                    i = i.replace("\n", "")
                    counter += 1
                    if counter > 1:
                        i_temp_dict = {"heading": i_temp_list[0],
                                       "sequence": "".join(i_temp_list[1:])}
                        self.genes.append(i_temp_dict)
                    i_temp_list = []
                    i_temp_list.append(i)
                else:
                    i = i.replace("\n", "")
                    i_temp_list.append(i)
            i_temp_dict = {"heading": i_temp_list[0],
                           "sequence": "".join(i_temp_list[1:])}
            self.genes.append(i_temp_dict)
            tax_name_fasta_find = tax_name_fasta_re_comp.findall(self.genes[0]["heading"])
            self.ref_name = tax_name_fasta_find[0].replace("OS=", "")
            for i in self.genes:
                prot_id_find = prot_id_re_comp.findall(i["heading"])
                yeast_gene_id_find = yeast_gene_id_re_comp.findall(i["heading"])
                GN_gene_id_find = GN_gene_id_re_comp.findall(i["heading"])
                prot_id_str = prot_id_find[0].replace("|", "")
                yeast_gene_id_str = yeast_gene_id_find[0].replace("_YEAST", "")
                GN_gene_id_str = GN_gene_id_find[0].replace("GN=", "")
                description_str = re.sub("\>\w+(\|\w+){2}\s", "", i["heading"])
                description_str = re.sub("\sOS\=.+", "", description_str)
                i["prot_id"] = prot_id_str
                i["yeast_gene_id"] = yeast_gene_id_str
                i["GN_gene_id"] = GN_gene_id_str
                i["description"] = description_str

    def parse_xmls(self,
                   in_file_names):
        """Return Genome.query_species (list of strs) and appends
        Genomes.genes with key:value entries of organisms that have orthologs
        for each gene from parsed (list) of OrthoXMLs. The file list is
        supposed to work with glob.

        Args:
            in_file_names (list of strs): OrthoXML files list to parse
        """
        counter = 0
        for i in in_file_names:
            print "\nprocessing {0}".format(i)
            with open(i, "r") as xml_fin:
                tree = et.parse(xml_fin)
            root = tree.getroot()
            if self.ref_name == str(root[2].attrib["name"])[:-1]:
                ref_species_tax_name = root[2].attrib["name"]
                query_species_tax_name = root[1].attrib["name"]
                query_species_uniprot_link = root[1][1][0].attrib["link"]
                query_species_orthologs = root[2][0][0]
            else:
                ref_species_tax_name = root[1].attrib["name"]
                query_species_tax_name = root[2].attrib["name"]
                query_species_uniprot_link = root[2][1][0].attrib["link"]
                query_species_orthologs = root[1][0][0]
            self.query_species.append(str(query_species_tax_name)[:-1])
            for ii in self.genes:
                prog_perc(ii, self.genes)
                for iii in query_species_orthologs[:]:
                    if ii["prot_id"] == iii.attrib["protId"]:
                        if "orthologs" in ii:
                            ii["orthologs"]["organism"].\
                            update({str(query_species_tax_name)[:-1]:
                                   {"prot_id": str(iii.attrib["protId"]),
                                    "gene_id": str(iii.attrib["geneId"])}})
                        else:
                            ii["orthologs"] = {"organism":
                                              {str(query_species_tax_name)[:-1]:
                                              {"prot_id": str(iii.attrib["protId"]),
                                               "gene_id": str(iii.attrib["geneId"])}}}
        self.query_species = tuple(self.query_species)

    def parse_orthologous_groups_csv(self,
                                     in_file_name,
                                     in_col_name):
        """Return Genome.query_species (list of strs) and appends
        Genomes.genes with key:value entries of organisms that have orthologs
        for each gene from parsed ortho-finder csv.

        Args:
            in_file_name (str): (csv) file name to parse
            in_col_name (str): column name in <in_file_name> respresentig the
            reference organism. Will NOT be inculuded in Genome.query_species
            or as +/- sign in Genome.gene_profiles.
        """
        clean_dict = {}
        out_list = []
        temp_dict_list_1 = []
        self.orthologous_groups_df = pd.read_csv(in_file_name,
                                                 sep = "\t",
                                                 index_col = 0)
        for i in self.orthologous_groups_df.columns:
            if i == in_col_name:
                pass
            else:
                self.query_species.append(str(i).split(".")[:-1][0])
        print "parsing reference organism ortho groups...".format()
        for i in self.orthologous_groups_df[in_col_name].iteritems():
            temp_dict_list_1.append({i[0]: map(lambda x: x.split("|"),
                                               str(i[1]).split(", "))})
        print "processing reference organism ortho groups..".format()
        for i in temp_dict_list_1:
            prog_perc(i, temp_dict_list_1)
            temp_ortho_group_str = str(i.keys()[0])
            temp_prot_id_list = []
            temp_yeast_gene_id_list = []
            for ii in i.values():
                for iii in ii:
                    if len(iii) > 1:
                        temp_prot_id_list.append(iii[1])
                        temp_yeast_gene_id_list.append(iii[2].split("_")[0])
                    else:
                        temp_prot_id_list.append("not_found")
                        temp_yeast_gene_id_list.append("not_found")
            temp_dict_1 = {"ortho_group_number": temp_ortho_group_str,
                           "prot_id": temp_prot_id_list,
                           "yeast_gene_id": temp_yeast_gene_id_list}
            self.orthologous_groups_dict.append(temp_dict_1)
        print "appending reference organism genes db...".format()
        for i in self.genes:
            prog_perc(i, self.genes)
            for ii in self.orthologous_groups_dict:
                if i["prot_id"] in ii["prot_id"]:
                    org_temp_str_list = []
                    org_temp_utf_list =  self.orthologous_groups_df.columns[self.orthologous_groups_df.ix[ii["ortho_group_number"]].notnull()]
                    for e in org_temp_utf_list:
                        org_temp_str_list.append(str(e).split(".")[0])
                    i["ortho_group"] =  ii["ortho_group_number"]
                    i["orthologs"] = {"organism":
                                     {e:
                                     {"ortho_group_number": ii["ortho_group_number"]}\
                                     for e in org_temp_str_list}}

    def no_orthologs_genes_remover(self):
        """Return Genome.empty_genes (list of dicts) and Genome.ortho_genes
        (list of dicts) by iterating over Genome.genes (list of dicts).
        Discrimination based on presence/absence of <orthologs> key.
        """
        for i in self.genes:
            if "orthologs" not in i:
                self.empty_genes.append(i)
            else:
                self.ortho_genes.append(i)

    def profiler(self, id_type = "prot_id"):
        """Return Genome.gene_profiles (list of tuples) by iterating over
        Genome.query_species (tuple of str) Genome.ortho_genes (list of dicts)
        <orthologs><organism> values. +/- signs sequence resembles
        Genome.query_species order.
        """
        for i in self.ortho_genes:
            temp_list = [i[id_type]]
            for ii in self.query_species:
                if ii in i["orthologs"]["organism"]:
                    temp_list.append("+")
                else:
                    temp_list.append("-")
            self.gene_profiles.append(tuple(temp_list))

class Ortho_Interactions():
    """Holds data about gene interactions array extracted from (csv) file.
    Merges these data with Genome.gene_profiles (list of tuples) and
    Genome.genes (list of dicts) selected values.

    Attribs:
        query_species_Genome (tuple of strs): passed from Genome

        genes_Genome (list of dicts): passed from Genome

        gene_profiles_Genome (list of tuples): passed from Genome
    """
    def __init__(self,
                 query_species_Genome,
                 genes_Genome,
                 gene_profiles_Genome):
        self.query_species_inter = query_species_Genome
        self.genes_inter = genes_Genome
        self.gene_profiles_inter = gene_profiles_Genome

    def parse_sgadata(self,
                      in_file_name,
                      p_value = float(0.05),
                      DMF_type = "neutral"):
        """Return Ortho_Interactions.interact_df (pandas.DataFrame) from
        parsed <csv> file. The minimal filtration is based of a given p-value
        and presence of DMF value. Further filtration results in DMF
        higher/lower than both SMFs.

        Args:
            in_file_name (str): name of (csv) file to parse

            p_value (float): maximum p-value for filtering

            DMF_type (str): positive -> DMF > both SMFs
                            negative -> DMF < both SMFs
                            neutral  -> DMF not <None> (default)
                            raw      -> no filter
        """
        print "reading in interactions csv...".format()
        csv_df = pd.read_csv(in_file_name)
        positive_DMF_bool = (csv_df["DMF"] > csv_df["Query_SMF"]) &\
                            (csv_df["DMF"] > csv_df["Array_SMF"]) &\
                            (csv_df["p-value"] <= p_value)
        negative_DMF_bool = (csv_df["DMF"] < csv_df["Query_SMF"]) &\
                            (csv_df["DMF"] < csv_df["Array_SMF"]) &\
                            (csv_df["p-value"] <= p_value)
        neutral_DMF_bool = (csv_df["DMF"].isnull() == False) &\
                           (csv_df["p-value"] <= p_value)
        print "selecting data...".format()
        if DMF_type == "positive":
            self.interact_df = csv_df[positive_DMF_bool]
        elif DMF_type == "negative":
            self.interact_df = csv_df[negative_DMF_bool]
        elif DMF_type == "neutral":
            self.interact_df = csv_df[neutral_DMF_bool]
        elif DMF_type == "raw":
            self.interact_df = csv_df
        else:
            pass

    def df_profiles_and_score_appender(self,
                                       profiles_df = True):
        """Return Ortho_Interactions.interact_df appended by concatenated
        Genome.gene_profiles (list of tuples), Genome.gene_profiles similarity
        score (float), Genome.genes(list of dicts) gene descriptors. Optionally
        appends with Genome.gene_profiles array browsable by organism's name.

        Args:
            profiles_df (bool): appends with Genome.gene_profiles array when
            <True> (default). Removes <None> rows
        """
        prof_score_temp_list = [df_qa_names_2_prof_score(ii,
                                                         self.gene_profiles_inter)
                                for ii in [[getattr(i, "Query_gene_name"),
                                            getattr(i, "Array_gene_name")]
                                for i in self.interact_df.itertuples()]]
        conc_qa_prof_temp_list = [[gene_profile_finder_by_name(ii[0],
                                   self.gene_profiles_inter,
                                   conc = True),
                                   gene_profile_finder_by_name(ii[1],
                                   self.gene_profiles_inter,
                                   conc = True)]
                                  for ii in [[getattr(i, "Query_gene_name"),
                                              getattr(i, "Array_gene_name")]
                                  for i in self.interact_df.itertuples()]]
        q_gene_head_temp_list = [gene_finder_by_attrib("GN_gene_id",
                                                       getattr(i, "Query_gene_name"),
                                                       "description",
                                                       self.genes_inter)
                                 for i in self.interact_df.itertuples()]
        a_gene_head_temp_list = [gene_finder_by_attrib("GN_gene_id",
                                                       getattr(i, "Array_gene_name"),
                                                       "description",
                                                       self.genes_inter)
                                 for i in self.interact_df.itertuples()]
        prof_score_temp_df = pd.DataFrame(prof_score_temp_list,
                                          index = self.interact_df.index,
                                          columns = ["Profiles_similarity_score"])
        conc_q_a_prof_temp_df = pd.DataFrame(conc_qa_prof_temp_list,
                                             index = self.interact_df.index,
                                             columns = ["Query_gene_profile", "Array_gene_profile"])
        q_gene_head_temp_df = pd.DataFrame(q_gene_head_temp_list,
                                           index = self.interact_df.index,
                                           columns = ["Query_gene_description"])
        a_gene_head_temp_df = pd.DataFrame(a_gene_head_temp_list,
                                           index = self.interact_df.index,
                                           columns = ["Array_gene_description"])
        self.interact_df = pd.concat([self.interact_df,
                                      prof_score_temp_df,
                                      conc_q_a_prof_temp_df,
                                      q_gene_head_temp_df,
                                      a_gene_head_temp_df],
                                      axis = 1)
        if profiles_df == True:
            cols_query_temp_list = ["Query_gene_name"] + list(self.query_species_inter)
            cols_array_temp_list = ["Array_gene_name"] + list(self.query_species_inter)
            sep_prof_temp_df = pd.DataFrame(self.gene_profiles_inter,
                                            columns = cols_query_temp_list)
            self.interact_df = pd.merge(self.interact_df,
                                        sep_prof_temp_df,
                                        on = "Query_gene_name")
            sep_prof_temp_df.columns = cols_array_temp_list
            self.interact_df = pd.merge(self.interact_df,
                                        sep_prof_temp_df,
                                        on = "Array_gene_name",
                                        suffixes = ("_query", "_array"))
        else:
            pass

class Ortho_Stats:
    """Calculates and holds data about interactions array statistical
    properties.
    """
    def __init__(self,
                 query_species_Genome,
                 gene_profiles_Genome,
                 inter_df_Ortho_Interactions):
        self.query_species_stats = query_species_Genome
        self.inter_df_stats = inter_df_Ortho_Interactions
        self.gene_profiles_stats = gene_profiles_Genome
        self.tot_inter_num = None
        self.DMF_positive_num = None
        self.DMF_negative_num = None
        self.sim_prof_num = None
        self.e_value = None
        self.res_val = None
        self.sim_perm_val = []
        self.unsim_perm_val = []
        self.mir_perm_val = []

    def df_selector(self,
                    DMF_pos = True):
        """Return pandas DataFrame selected to chosen DMF type (bool).

        Args:
            DMF_pos (bool): selects only positive DMF type. Default.
        """
        positive_DMF_bool = (self.inter_df_stats["DMF"] >\
                             self.inter_df_stats["Query_SMF"]) &\
                            (self.inter_df_stats["DMF"] >\
                             self.inter_df_stats["Array_SMF"])
        negative_DMF_bool = (self.inter_df_stats["DMF"] <\
                             self.inter_df_stats["Query_SMF"]) &\
                            (self.inter_df_stats["DMF"] <\
                             self.inter_df_stats["Array_SMF"])
        if DMF_pos == True:
            self.inter_df_stats = self.inter_df_stats[positive_DMF_bool]
        else:
            self.inter_df_stats = self.inter_df_stats[negative_DMF_bool]

    def prof_perm(self,
                  e_value,
                  in_prof_sim_lev):
        """Return lists of number of different types of profiles scores, each
        generated from shuffled pandas DataFrame.

        Args:
            e_value (int): number of times to shuffle the pandas DataFrame
            in_prof_sim_lev(int): treshold for assuming profiles as similar or
            not
        """
        for i in range(e_value):
            temp_score_list = []
            q_temp_df = self.inter_df_stats[["Query_gene_name", "Query_gene_profile"]]
            a_temp_df = self.inter_df_stats[["Array_gene_name", "Array_gene_profile"]]
            q_temp_perm_df = q_temp_df.sample(len(q_temp_df))
            a_temp_perm_df = a_temp_df.sample(len(a_temp_df))
            q_temp_perm_df.index = range(len(q_temp_perm_df))
            a_temp_perm_df.index = range(len(a_temp_perm_df))
            qa_temp_perm_df = pd.concat([q_temp_perm_df, a_temp_perm_df], axis=1)
            for ii in qa_temp_perm_df.itertuples():
                temp_score_list.append(df_qa_names_2_prof_score([getattr(ii, "Query_gene_name"),
                                                                 getattr(ii, "Array_gene_name")],
                                                                self.gene_profiles_stats))
            temp_score_df = pd.DataFrame(temp_score_list,
                                         index=qa_temp_perm_df.index,
                                         columns=["Profiles_similarity_score"])
            qa_temp_perm_score_df = pd.concat([qa_temp_perm_df, temp_score_df],
                                               axis=1)
            sim_prof_bool = (qa_temp_perm_score_df["Profiles_similarity_score"] >=\
                             in_prof_sim_lev)
            unsim_prof_bool = (qa_temp_perm_score_df["Profiles_similarity_score"] <\
                               in_prof_sim_lev) &\
                              (qa_temp_perm_score_df["Profiles_similarity_score"] > 0)
            mir_prof_bool = (qa_temp_perm_score_df["Profiles_similarity_score"] == 0)
            sim_prof_perm_num = len(qa_temp_perm_score_df[sim_prof_bool])
            unsim_prof_perm_num = len(qa_temp_perm_score_df[unsim_prof_bool])
            mir_prof_perm_num = len(qa_temp_perm_score_df[mir_prof_bool])
            self.sim_perm_val.append(sim_prof_perm_num)
            self.unsim_perm_val.append(unsim_prof_perm_num)
            self.mir_perm_val.append(mir_prof_perm_num)

    def df_num_prop(self,
                    in_prof_sim_lev):
        """Return Ortho_Stats.tot_inter_num (int),
        Ortho_Stats.DMF_positive_num (int),
        Ortho_Stats.DMF_negative_num (int),
        Ortho_Stats.sim_prof_num (int).

        Args:
            in_prof_sim_lev (int): defines minimal Genome.gene_profiles in
            Ortho_Stats.inter_df_stats similarity treshold
        """
        positive_DMF_bool = (self.inter_df_stats["DMF"] >\
                             self.inter_df_stats["Query_SMF"]) &\
                            (self.inter_df_stats["DMF"] >\
                             self.inter_df_stats["Array_SMF"])
        negative_DMF_bool = (self.inter_df_stats["DMF"] <\
                             self.inter_df_stats["Query_SMF"]) &\
                            (self.inter_df_stats["DMF"] <\
                             self.inter_df_stats["Array_SMF"])
        sim_prof_bool = (self.inter_df_stats["Profiles_similarity_score"] >=\
                         in_prof_sim_lev)
        self.tot_inter_num = len(self.inter_df_stats)
        self.DMF_positive_num = len(self.inter_df_stats[positive_DMF_bool])
        self.DMF_negative_num = len(self.inter_df_stats[negative_DMF_bool])
        self.sim_prof_num = len(self.inter_df_stats[sim_prof_bool])
        self.res_val = len(self.inter_df_stats[positive_DMF_bool & sim_prof_bool])

    def e_val_calc(self):
        """Return Ortho_Stats.e_value (int) which is an expected number of
        interactions with positive DMF and similar gene profiles by chance.
        """
        self.e_value = (self.DMF_positive_num * self.sim_prof_num) /\
                        self.tot_inter_num

def main():
    pass

if __name__ == "__main__":
    main()
