#! /usr/bin/env python

__author__ = "Dariusz Izak"

import lxml.etree as et
import pandas as pd
import re
import math
import argparse
import glob
import sys
import requests as rq
import numpy as np
import pathos.multiprocessing as ptmp
import networkx as nx
import matplotlib.pyplot as plt
import jinja2 as jj2
import time

def perc_prog(in_item,
              in_iterbl):
    """Display progress of iterable as percent. Uses carriage return. Works in
    the terminal only.

    Args:
        in_item: current iteration element
        in_iterbl: iterable
    """
    pos = in_iterbl.index(in_item)
    sys.stdout.write("{0}%\r".format(pos * 100 /\
                                     len(in_iterbl)))
    sys.stdout.flush()

def sign_prog(in_item,
              in_iterbl,
              in_size = 50,
              in_sign = "|"):
    """Display progress of iterable as bar. Iterable must support built-in
    index method.

    Args:
        in_item: current iteration element
        in_iterbl: iterable
        in_size: total number of signs in completed bar. Equals number of
        iterations when argument higher than total number of iterations.
    """
    if in_size > len(in_iterbl):
        in_size = len(in_iterbl)
    else:
        pass
    tot_len = len(in_iterbl)
    sign_size = float(tot_len / in_size)
    if in_iterbl.index(in_item) % sign_size == 0:
        sys.stdout.write(in_sign)
        sys.stdout.flush()

def sign_prog_counter(in_counter,
                      in_iterbl,
                      in_size = 50,
                      in_sign = "|"):
    """Display progress of iterable as bar. For each iteration a counter must
    be provided and then passed as an argument.

    Args:
        in_counter: current iteration element
        in_iterbl: iterable
        in_size: total number of signs in completed bar. Equals number of
        iterations when argument higher than total number of iterations.
    Examples:
        >>> counter = 0
        >>> for i in range(100):
                counter += 1
                sign_prog_counter(counter, range(100))
        ||||||||||||||||||||||||||||||||||||||||||||||||||
    """
    if in_size > len(in_iterbl):
        in_size = len(in_iterbl)
    else:
        pass
    tot_len = len(in_iterbl)
    sign_size = float(tot_len / in_size)
    if in_counter % sign_size == 0:
        sys.stdout.write(in_sign)
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
    for i in in_all_gene_profiles:
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
        in_gene_profile_1 (numpy.array): the first profile to compare. MUST be
        generated from list. MUST NOT contain gene name.
        in_gene_profile_2 (numpy.array): the second profile to compare. MUST be
        generated from list. MUST NOT contain gene name.
    """
    return (in_gene_profile_1 == in_gene_profile_2).sum()

def df_based_profiles_scorer(in_df,
                             prof_1_col_name,
                             prof_2_col_name,
                             score_col_name):
    temp_score_list = []
    for i in in_df.itertuples():
        prof_1 = np.array(list(getattr(i, prof_1_col_name)))
        prof_2 = np.array(list(getattr(i, prof_2_col_name)))
        temp_score_list.append(simple_profiles_scorer(prof_1, prof_2))
    temp_score_df = pd.DataFrame(temp_score_list,
                                 index = in_df.index,
                                 columns = [score_col_name])
    in_df = pd.concat([in_df,
                       temp_score_df],
                      axis = 1)
    return in_df

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
        self.KO_list = []
        self.KO_df = None
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
                sign_prog(ii, range(len(self.genes)))
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
        print "\nparsing reference organism ortho groups...".format()
        for i in self.orthologous_groups_df[in_col_name].iteritems():
            temp_dict_list_1.append({i[0]: map(lambda x: x.split("|"),
                                               str(i[1]).split(", "))})
        print "\nprocessing reference organism ortho groups..".format()
        for i in temp_dict_list_1:
            sign_prog(i, temp_dict_list_1)
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
        print "\nappending reference organism genes db...".format()
        for i in self.genes:
            sign_prog(i, self.genes)
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

    def parse_KO_db(self,
                    in_file_name):
        """Return Genome.KO_list (list of dicts) which contains information from
        the file downloaded by KEGG_API.get_ortho_db_entries.

        Args:
            in_file_name (str): file name to parse
        """
        with open(in_file_name, "r") as fin:
            file_str = fin.read()
            entries_list = file_str.split("///")
        def f(i):
            entry_dict = {}
            pathway_dict = {}
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
            genes_blk_comp = re.compile("GENES.+\n^\s+\w{3}:\s.+^\w", re.DOTALL|re.MULTILINE)
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
        self.KO_list = ptmp.ProcessingPool().map(f, entries_list)

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

    def profiler(self,
                 id_type = "prot_id"):
        """Return Genome.gene_profiles (list of tuples) by iterating over
        Genome.query_species (tuple of str) Genome.ortho_genes (list of dicts)
        <orthologs><organism> values. +/- signs sequence resembles
        Genome.query_species order.

        Args:
            id_type (str): ID type to use as the first element in the profile
            (list). Must be congruous with ID type in Genome.genes
        """
        for i in self.ortho_genes:
            temp_arr = np.array([i[id_type]])
            for ii in self.query_species:
                if ii in i["orthologs"]["organism"]:
                    temp_arr = np.append(temp_arr, "+")
                else:
                    temp_arr = np.append(temp_arr, "-")
            self.gene_profiles.append(temp_arr)

    def KO_list_profiler(self,
                         species_ids,
                         remove_empty = True,
                         upperize_ids = True,
                         profile_list = False,
                         KO_list_2_df = True):
        """Return Genome.KO_list (list of dict) of Genome.KO_df (pandas.DataFrame)
        appended with profiles (list of str or str).

        Args:
            species_ids (list of str): KEGG's IDs (3-letters) of reference
            species upon which are built.
            remove_empty (bool): remove inplace None types from the species_ids
            (list) <True> (default)
            upperize_ids (bool): make the items from the species_ids (list)
            upper-case as it is in the Genome.KO_list orgs key when <True>
            (default)
            profile_list (bool): return each profile as the list of separate
            "+" or "-" when <True> or as one str when <False> (default)
            KO_list_2_df (bool): convert Genome.KO_list to pandas.DataFrame
        """
        if remove_empty == True:
            species_ids = [i for i in species_ids if i != None]
        if upperize_ids == True:
            species_ids = [i.upper() for i in species_ids]
        for i in self.KO_list:
            if "orgs" in i.keys():
                profile = ["+" if ii in i["orgs"] else "-" for ii in species_ids]
                if profile_list == False:
                    profile = "".join(profile)
                else:
                    pass
                i["profile"] = profile
            else:
                pass
        if KO_list_2_df == True:
            self.KO_df = pd.DataFrame(self.KO_list)

class Ortho_Interactions:
    """Holds data about gene interactions array extracted from (csv) file.
    Merges these data with Genome.gene_profiles (list of tuples) and
    Genome.genes (list of dicts) selected values or Genome.KO_df.

    Attribs:
        query_species_Genome (tuple of strs): passed from Genome. Set to <None>
        if using data from KEGG
        genes_Genome (list of dicts): passed from Genome. Set to <None> if using
        data from KEGG
        gene_profiles_Genome (list of tuples): passed from Genome. Set to <None>
        if using data from KEGG
        ORF_KO_df (pandas.DataFrame): passed from KEGG_API. Consists of 2
        columns - ORF name and KO orthology group ID
        interact_df (pandas.DataFrame): holds data about interactions from
        parsed csv file. Can be appended with
        Ortho_Interactions.gen_based_appender or
        Ortho_Interactions.ko_based_appender
    """
    def __init__(self,
                 query_species_Genome,
                 genes_Genome,
                 gene_profiles_Genome,
                 KO_df_Genome,
                 org_ortho_db_X_ref_df_KEGG_API):
        self.query_species_inter = query_species_Genome
        self.genes_inter = genes_Genome
        self.gene_profiles_inter = gene_profiles_Genome
        self.KO_df_inter = KO_df_Genome
        self.ORF_KO_df = org_ortho_db_X_ref_df_KEGG_API
        self.interact_df = None
        self.bio_proc_df = None

    def parse_sgadata(self,
                      in_file_name,
                      p_value = float(0.05),
                      DMF_type = "neutral",
                      excel = False,
                      in_sep = ","):
        """Return Ortho_Interactions.interact_df (pandas.DataFrame) from
        parsed <csv> file. The minimal filtration is based of a given p-value
        and presence of DMF value. Further filtration results in DMF
        higher/lower than both SMFs.

        Args:
            in_file_name (str): name of file to parse
            p_value (float): maximum p-value for filtering
            DMF_type (str): positive -> DMF > both SMFs
                            negative -> DMF < both SMFs
                            neutral  -> DMF not <None> (default)
                            raw      -> no filter
            excel (bool): pandas.read_excel when <True>. pandas.read_csv when
            <False> (default).
            in_sep (str): separator for pandas.read_csv method
        """
        print "\nreading in interactions csv...".format()
        if excel == False:
            sga_df = pd.read_csv(in_file_name, sep = in_sep)
        else:
            sga_df = pd.read_excel(in_file_name)
        positive_DMF_bool = (sga_df["DMF"] > sga_df["Query_SMF"]) &\
                            (sga_df["DMF"] > sga_df["Array_SMF"]) &\
                            (sga_df["p-value"] <= p_value)
        negative_DMF_bool = (sga_df["DMF"] < sga_df["Query_SMF"]) &\
                            (sga_df["DMF"] < sga_df["Array_SMF"]) &\
                            (sga_df["p-value"] <= p_value)
        neutral_DMF_bool = (sga_df["DMF"].isnull() == False) &\
                           (sga_df["p-value"] <= p_value)
        print "\nselecting data...".format()
        if DMF_type == "positive":
            self.interact_df = sga_df[positive_DMF_bool]
        elif DMF_type == "negative":
            self.interact_df = sga_df[negative_DMF_bool]
        elif DMF_type == "neutral":
            self.interact_df = sga_df[neutral_DMF_bool]
        elif DMF_type == "raw":
            self.interact_df = sga_df
        else:
            pass

    def parse_bioprocesses(self,
                           in_file_name,
                           excel = False,
                           in_sep = ","):
        """Return Ortho_Interactions.bio_proc_df (pandas.DataFrame) from parsed
        <csv> or <xls> file.

        Args:
            in_file_name (str): name of file to parse
            excel (bool): pandas.read_excel when <True>. pandas.read_csv when
            <False> (default).
            in_sep (str): separator for pandas.read_csv method
        """
        if excel == False:
            self.bio_proc_df = pd.read_csv(in_file_name, sep = in_sep)
        else:
            self.bio_proc_df = pd.read_excel(in_file_name)

    def gen_based_appender(self,
                           bio_proc = True,
                           profiles_df = True):
        """Return Ortho_Interactions.interact_df appended by concatenated
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
        for i in self.interact_df.itertuples():
            qa_attrib_temp_list.append([getattr(i, "Query_gene_name"),
                                        getattr(i, "Array_gene_name")])
        print "\nscoring profiles similarity...".format()
        prof_score_temp_list = ptmp.ProcessingPool().map(lambda x: df_qa_names_2_prof_score(x,
                                                                                            self.gene_profiles_inter),
                                                         qa_attrib_temp_list)
        print "\nconcatenating profiles...".format()
        conc_qa_prof_temp_list = ptmp.ProcessingPool().map(lambda x: [gene_profile_finder_by_name(x[0],
                                                                                                  self.gene_profiles_inter,
                                                                                                  conc = True),
                                                                      gene_profile_finder_by_name(x[1],
                                                                                                  self.gene_profiles_inter,
                                                                                                  conc = True)],
                                                           qa_attrib_temp_list)
        print "\npreparing descriptors of query genes...".format()
        for i in self.interact_df.itertuples():
            q_gene_name_temp_list.append(getattr(i, "Query_gene_name"))
        q_gene_head_temp_list = ptmp.ProcessingPool().map(lambda x: gene_finder_by_attrib("GN_gene_id",
                                                                                          x,
                                                                                          "description",
                                                                                          self.genes_inter),
                                                          q_gene_name_temp_list)
        print "\npreparing descriptors of array genes...".format()
        for i in self.interact_df.itertuples():
            a_gene_name_temp_list.append(getattr(i, "Array_gene_name"))
        a_gene_head_temp_list = ptmp.ProcessingPool().map(lambda x: gene_finder_by_attrib("GN_gene_id",
                                                                                          x,
                                                                                          "description",
                                                                                          self.genes_inter),
                                                          a_gene_name_temp_list)
        print "\ncreating temporary dataframes...".format()
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
        print "\nconcatenating dataframes...".format()
        self.interact_df = pd.concat([self.interact_df,
                                      prof_score_temp_df,
                                      conc_q_a_prof_temp_df,
                                      q_gene_head_temp_df,
                                      a_gene_head_temp_df],
                                      axis = 1)
        if bio_proc == True:
            print "\nappending with bioprocesses info...".format()
            self.interact_df = pd.merge(self.interact_df,
                                        self.bio_proc_df,
                                        left_on = "Query_gene_name",
                                        right_on = "Gene_name",
                                        how = "left")
            self.interact_df = pd.merge(self.interact_df,
                                        self.bio_proc_df,
                                        left_on = "Array_gene_name",
                                        right_on = "Gene_name",
                                        how = "left",
                                        suffixes=("_query", "_array"))
            self.interact_df.drop(["Gene_name_query", "Gene_name_array"],
                                  axis = 1,
                                  inplace = True)
            for i in self.interact_df.itertuples():
                if getattr(i, "Process_query") == getattr(i, "Process_array"):
                    if getattr(i, "Process_query") == "unknown" or\
                       getattr(i, "Process_query") == "unknown":
                        bio_proc_temp_list.append("unknown")
                    else:
                        bio_proc_temp_list.append("identical")
                else:
                    bio_proc_temp_list.append("different")
            bio_proc_temp_df = pd.DataFrame(bio_proc_temp_list,
                                            index = self.interact_df.index,
                                            columns = ["Bioprocesses_similarity"])
            self.interact_df = pd.concat([self.interact_df,
                                          bio_proc_temp_df],
                                         axis = 1)
        else:
            pass
        if profiles_df == True:
            print "\nappending with sign-per-column profiles...".format()
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

    def KO_based_appender(self):
        temp_score_list = []
        self.interact_df = pd.merge(self.interact_df,
                                    self.ORF_KO_df,
                                    left_on = "Query_ORF",
                                    right_on = "ORF_id",
                                    how = "left")
        self.interact_df = pd.merge(self.interact_df,
                                    self.ORF_KO_df,
                                    left_on = "Array_ORF",
                                    right_on = "ORF_id",
                                    how = "left",
                                    suffixes = ("_query", "_array"))
        self.interact_df.drop(["ORF_id_query", "ORF_id_array"],
                              axis = 1,
                              inplace = True)
        self.interact_df.dropna(inplace = True)
        self.interact_df = pd.merge(self.interact_df,
                                    self.KO_df_inter,
                                    left_on = "kegg_id_query",
                                    right_on = "entry",
                                    how = "left")
        self.interact_df = pd.merge(self.interact_df,
                                    self.KO_df_inter,
                                    left_on = "kegg_id_array",
                                    right_on = "entry",
                                    how = "left",
                                    suffixes=('_query', '_array'))
        self.interact_df.dropna(inplace = True)
        self.interact_df.rename(columns = {"profile_query": "Query_gene_profile",
                                           "profile_array": "Array_gene_profile"},
                                inplace = True)
        for i in self.interact_df.itertuples():
            prof_1 = np.array(list(getattr(i, "Query_gene_profile")))
            prof_2 = np.array(list(getattr(i, "Array_gene_profile")))
            temp_score_list.append(simple_profiles_scorer(prof_1,
                                                          prof_2))
        temp_score_df = pd.DataFrame(temp_score_list,
                                     index = self.interact_df.index,
                                     columns = ["Profiles_similarity_score"])
        self.interact_df = pd.concat([self.interact_df,
                                     temp_score_df],
                                     axis = 1)

    def bio_proc_appender(self):
        bio_proc_temp_list = []
        self.interact_df = pd.merge(self.interact_df,
                                    self.bio_proc_df,
                                    left_on = "Query_gene_name",
                                    right_on = "Gene_name",
                                    how = "left")
        self.interact_df = pd.merge(self.interact_df,
                                    self.bio_proc_df,
                                    left_on = "Array_gene_name",
                                    right_on = "Gene_name",
                                    how = "left",
                                    suffixes=("_query", "_array"))
        self.interact_df.drop(["Gene_name_query", "Gene_name_array"],
                              axis = 1,
                              inplace = True)
        for i in self.interact_df.itertuples():
            if getattr(i, "Process_query") == getattr(i, "Process_array"):
                if getattr(i, "Process_query") == "unknown" or\
                   getattr(i, "Process_query") == "unknown":
                    bio_proc_temp_list.append("unknown")
                else:
                    bio_proc_temp_list.append("identical")
            else:
                bio_proc_temp_list.append("different")
        bio_proc_temp_df = pd.DataFrame(bio_proc_temp_list,
                                        index = self.interact_df.index,
                                        columns = ["Bioprocesses_similarity"])
        self.interact_df = pd.concat([self.interact_df,
                                      bio_proc_temp_df],
                                     axis = 1)

    def inter_df_read(self,
                      in_file_name,
                      in_sep = "\t"):
        self.interact_df = pd.read_csv(in_file_name,
                                       sep = in_sep)

    def inter_df_save(self,
                      out_file_name,
                      in_sep = "\t"):
        self.interact_df.to_csv("{0}.csv".format(out_file_name),
                                sep = in_sep,
                                index = False)

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
        self.num_prop_res = None
        self.e_value = None
        self.perm_results = None
        self.prof_arr_perm_res_avg = None
        self.filters_used = "No filters"

    def df_selector(self,
                    DMF = None,
                    SMF_below_one = True,
                    inter_score_min = None,
                    inter_score_max = None,
                    no_flat_plus = False,
                    no_flat_minus = False,
                    process = None,
                    profiles = None,
                    prof_sim_lev = None):
        """Return filtered Ortho_Stats.interact_df_stats passed from
        Ortho_Interactions (pandas.DataFrame). For each filter, type <None> to
        omit.

        Args:
            DMF (str): selects DMF type. Possible: <positive>, <negative> or
            <None> (omit filter). Default: <None>
            inter_score_min (float): selects minimum Genetic interactions Score.
            Default <None>
            inter_score_max (float): selects minimum Genetic interactions Score.
            Default <None>
            no_flat (bool): eliminates mirror profiles. Default <False>
            process (str): selects bioprocesses similarity. Possible: <identical>,
            "different" or <None>. Default <None>
            profiles (str): selects similar or dissimilar profiles. Possible
            <similar>, <unsimilar> or <None>. Similarity threshold MUST be
            specified with the prof_sim_lev arg if profiles != <None>.
            Default <None>.
            prof_sim_lev (int): defines profiles as similar of dissimilar
            when above or below this given value
        """
        self.filters_used = []
        positive_DMF_bool = (self.inter_df_stats["DMF"] >\
                             self.inter_df_stats["Query_SMF"]) &\
                            (self.inter_df_stats["DMF"] >\
                             self.inter_df_stats["Array_SMF"])
        negative_DMF_bool = (self.inter_df_stats["DMF"] <\
                             self.inter_df_stats["Query_SMF"]) &\
                            (self.inter_df_stats["DMF"] <\
                             self.inter_df_stats["Array_SMF"])
        SMF_below_one_bool = (self.inter_df_stats["Query_SMF"] < 1.0) &\
                             (self.inter_df_stats["Array_SMF"] < 1.0)
        inter_score_max_bool = (self.inter_df_stats["Genetic_interaction_score"] < inter_score_max)
        inter_score_min_bool = (self.inter_df_stats["Genetic_interaction_score"] > inter_score_min)
        no_flat_plu_q_bool = (self.inter_df_stats["Query_gene_profile"] !=\
                               "+" * len(self.query_species_stats))
        no_flat_min_q_bool = (self.inter_df_stats["Query_gene_profile"] !=\
                               "-" * len(self.query_species_stats))
        no_flat_plu_a_bool = (self.inter_df_stats["Array_gene_profile"] !=\
                               "+" * len(self.query_species_stats))
        no_flat_min_a_bool = (self.inter_df_stats["Array_gene_profile"] !=\
                               "-" * len(self.query_species_stats))
        iden_proc_bool = (self.inter_df_stats["Bioprocesses_similarity"] ==\
                           "identical")
        diff_proc_bool = (self.inter_df_stats["Bioprocesses_similarity"] ==\
                          "different")
        if profiles != None:
            sim_prof_bool = (self.inter_df_stats["Profiles_similarity_score"] >=\
                             prof_sim_lev)
            unsim_prof_bool = (self.inter_df_stats["Profiles_similarity_score"] <\
                               prof_sim_lev)
        else:
            pass
        if DMF == "positive":
            self.inter_df_stats = self.inter_df_stats[positive_DMF_bool]
            self.filters_used.append("DMF positive")
        elif DMF == "negative":
            self.inter_df_stats = self.inter_df_stats[negative_DMF_bool]
            self.filters_used.append("DMF negative")
        else:
            pass
        if SMF_below_one == True:
            self.inter_df_stats = self.inter_df_stats[SMF_below_one_bool]
            self.filters_used.append("SMF < 1.0")
        else:
            pass
        if isinstance(inter_score_max, float) == True:
            self.inter_df_stats = self.inter_df_stats[inter_score_max_bool]
            self.filters_used.append("Genetic interaction score < {0}".format(inter_score_max))
        else:
            pass
        if isinstance(inter_score_min, float) == True:
            self.inter_df_stats = self.inter_df_stats[inter_score_min_bool]
            self.filters_used.append("Genetic interaction score > {0}".format(inter_score_min))
        else:
            pass
        if no_flat_plus == True:
            self.inter_df_stats = self.inter_df_stats[no_flat_plu_q_bool]
            self.inter_df_stats = self.inter_df_stats[no_flat_plu_a_bool]
            self.filters_used.append("No plus-only (eg ++++++) profiles")
        else:
            pass
        if no_flat_minus == True:
            self.inter_df_stats = self.inter_df_stats[no_flat_min_q_bool]
            self.inter_df_stats = self.inter_df_stats[no_flat_min_a_bool]
            self.filters_used.append("No minus-only (eg ------) profiles")
        else:
            pass
        if process == "identical":
            self.inter_df_stats = self.inter_df_stats[iden_proc_bool]
            self.filters_used.append("Identical bioprocesses")
        elif process == "different":
            self.inter_df_stats = self.inter_df_stats[diff_proc_bool]
            self.filters_used.append("Different bioprocesses")
        else:
            pass
        if profiles == "similar":
            self.inter_df_stats = self.inter_df_stats[sim_prof_bool]
            self.filters_used.append("Similar profiles")
        elif profiles == "unsimilar":
            self.inter_df_stats = self.inter_df_stats[unsim_prof_bool]
            self.filters_used.append("Dissimilar profiles")
        else:
            pass

    def df_num_prop(self,
                    in_prof_sim_lev):
        """Return Ortho_Stats.tot_inter_num (int),
        Ortho_Stats.DMF_positive_num (int),
        Ortho_Stats.DMF_negative_num (int),
        Ortho_Stats.sim_prof_num (int).

        Args:
            in_prof_sim_lev (int): definges minimal Genome.gene_profiles in
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
        unsim_prof_bool = (self.inter_df_stats["Profiles_similarity_score"] <\
                           in_prof_sim_lev) &\
                          (self.inter_df_stats["Profiles_similarity_score"] > 0)
        mir_prof_bool = (self.inter_df_stats["Profiles_similarity_score"] == 0)
        self.num_prop_res = pd.Series({"total": len(self.inter_df_stats),
                                       "DMF_positive": len(self.inter_df_stats[positive_DMF_bool]),
                                       "DMF_negative": len(self.inter_df_stats[negative_DMF_bool]),
                                       "similar_profiles": len(self.inter_df_stats[sim_prof_bool]),
                                       "unsimilar_profiles": len(self.inter_df_stats[unsim_prof_bool]),
                                       "mirror_profiles": len(self.inter_df_stats[mir_prof_bool])})
        self.filters_used.append("Profiles similarity threshold: {0}".format(in_prof_sim_lev))

    def names_perm(self,
                   e_value,
                   in_prof_sim_lev):
        """Return pandas.DataFrame of number of different types of profiles scores, each
        generated from pandas DataFrame in which genes names were permuted. It is
        an equivalent of creating completely new, random network.

        Args:
            e_value (int): number of times to shuffle the pandas DataFrame
            in_prof_sim_lev(int): treshold for assuming profiles as similar or
            not
        """
        def f(in_iter):
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
            return {"similar": sim_prof_perm_num,
                    "unsimilar": unsim_prof_perm_num,
                    "mirror": mir_prof_perm_num,
                    "iteration": in_iter + 1}
        perm_results_temp_dict = ptmp.ProcessingPool().map(f, range(e_value))
        self.perm_results = pd.DataFrame(perm_results_temp_dict)

    def prof_cols_perm(self,
                       e_value,
                       in_prof_sim_lev):
        """Return pandas.DataFrame of number of different types of profiles scores,
        ech generated from pandas.DataFrame in which gene profiles were permuted but
        NOT the rest of the data. It is an equivalent of permuting parameters in
        the interactions network without changing the network's topology. Gene
        profiles are shuffled without any key.

        Args:
            e_value (int): number of times to shuffle the pandas DataFrame
            in_prof_sim_lev(int): treshold for assuming profiles as similar or
            not
        """
        q_sign_per_col_profs_cols = ["{0}_query".format(i) for i in self.query_species_stats]
        a_sign_per_col_profs_cols = ["{0}_array".format(i) for i in self.query_species_stats]
        def f(in_iter):
            temp_score_list = []
            q_prof_temp_df = self.inter_df_stats["Query_gene_profile"]
            a_prof_temp_df = self.inter_df_stats["Array_gene_profile"]
            drop_prof_temp_df = self.inter_df_stats.drop(["Query_gene_profile",
                                                          "Array_gene_profile",
                                                          "Profiles_similarity_score"] +\
                                                         q_sign_per_col_profs_cols +\
                                                         a_sign_per_col_profs_cols,
                                                         axis = 1)
            q_prof_perm_temp_df = q_prof_temp_df.sample(len(q_prof_temp_df))
            a_prof_perm_temp_df = a_prof_temp_df.sample(len(a_prof_temp_df))
            q_prof_perm_temp_df.index = drop_prof_temp_df.index
            a_prof_perm_temp_df.index = drop_prof_temp_df.index
            permuted_df = pd.concat([drop_prof_temp_df,
                                     q_prof_perm_temp_df,
                                     a_prof_perm_temp_df],
                                    axis = 1)
            for ii in permuted_df.itertuples():
                temp_score_list.append([simple_profiles_scorer(np.array(list(getattr(ii, "Query_gene_profile"))),
                                                               np.array(list(getattr(ii, "Array_gene_profile"))))])
            temp_score_df = pd.DataFrame(temp_score_list,
                                         index=permuted_df.index,
                                         columns=["Profiles_similarity_score"])
            permuted_profs_df = pd.concat([permuted_df,
                                          temp_score_df],
                                          axis = 1)
            sim_prof_bool = (permuted_profs_df["Profiles_similarity_score"] >=\
                             in_prof_sim_lev)
            unsim_prof_bool = (permuted_profs_df["Profiles_similarity_score"] <\
                               in_prof_sim_lev) &\
                              (permuted_profs_df["Profiles_similarity_score"] > 0)
            mir_prof_bool = (permuted_profs_df["Profiles_similarity_score"] == 0)
            sim_prof_perm_num = len(permuted_profs_df[sim_prof_bool])
            unsim_prof_perm_num = len(permuted_profs_df[unsim_prof_bool])
            mir_prof_perm_num = len(permuted_profs_df[mir_prof_bool])
            return {"similar": sim_prof_perm_num,
                    "unsimilar": unsim_prof_perm_num,
                    "mirror": mir_prof_perm_num,
                    "iteration": in_iter + 1}
        perm_results_temp_dict = ptmp.ProcessingPool().map(f, range(e_value))
        self.perm_results = pd.DataFrame(perm_results_temp_dict)

    def prof_arr_perm(self,
                      e_value,
                      in_prof_sim_lev):
        """Return a new Ortho_Stats.inter_df_stats which was stripped from
        gene_profiles data and then appended with gene_profiles again using
        a permuted gene_profiles list.

        Args:
            e_value (int): number of times to shuffle the pandas DataFrame
            in_prof_sim_lev(int): treshold for assuming profiles as similar or
            not
        """
        q_sign_per_col_profs_cols = ["{0}_query".format(i) for i in self.query_species_stats]
        a_sign_per_col_profs_cols = ["{0}_array".format(i) for i in self.query_species_stats]
        drop_prof_temp_df = self.inter_df_stats.drop(["Query_gene_profile",
                                                      "Array_gene_profile",
                                                      "Profiles_similarity_score"] +\
                                                     q_sign_per_col_profs_cols +\
                                                     a_sign_per_col_profs_cols,
                                                     axis = 1)
        def f(in_iter):
            gene_profs_perm_arr_list = []
            prof_score_temp_list = []
            qa_attrib_temp_list = []
            conc_qa_prof_temp_list = []
            gene_profs_names = [i[0] for i in self.gene_profiles_stats]
            gene_profs_profs = [i[1:] for i in self.gene_profiles_stats]
            gene_profs_names_ser = pd.Series(gene_profs_names)
            gene_profs_profs_ser = pd.Series(gene_profs_profs)
            gene_profs_names_ser_perm = gene_profs_names_ser.sample(len(gene_profs_names_ser))
            gene_profs_names_ser_perm.index = range(len(gene_profs_names_ser_perm))
            gene_profs_perm_df = pd.concat([gene_profs_names_ser_perm,
                                            gene_profs_profs_ser],
                                           axis = 1)
            gene_profs_perm_df.columns = ["perm_names", "profiles"]
            for i in gene_profs_perm_df.itertuples():
                name_arr = np.array(getattr(i, "perm_names"))
                full_arr = np.append(name_arr, getattr(i, "profiles"))
                gene_profs_perm_arr_list.append(full_arr)
            for i in drop_prof_temp_df.itertuples():
                qa_attrib_temp_list.append([getattr(i, "Query_gene_name"),
                                            getattr(i, "Array_gene_name")])
            for i in qa_attrib_temp_list:
                prof_score_temp_list.append(df_qa_names_2_prof_score(i,
                                                                     gene_profs_perm_arr_list))
                conc_qa_prof_temp_list.append([gene_profile_finder_by_name(i[0],
                                                                           gene_profs_perm_arr_list,
                                                                           conc = True),
                                               gene_profile_finder_by_name(i[1],
                                                                           gene_profs_perm_arr_list,
                                                                           conc = True)])
            prof_score_temp_df = pd.DataFrame(prof_score_temp_list,
                                              index = drop_prof_temp_df.index,
                                              columns = ["Profiles_similarity_score"])
            profs_pairs_temp_df = pd.DataFrame(conc_qa_prof_temp_list,
                                               index = drop_prof_temp_df.index,
                                               columns = ["Query_gene_profile", "Array_gene_profile"])
            permuted_df = pd.concat([drop_prof_temp_df,
                                     profs_pairs_temp_df,
                                     prof_score_temp_df],
                                     axis = 1)
            sim_prof_bool = (permuted_df["Profiles_similarity_score"] >=\
                             in_prof_sim_lev)
            unsim_prof_bool = (permuted_df["Profiles_similarity_score"] <\
                               in_prof_sim_lev) &\
                              (permuted_df["Profiles_similarity_score"] > 0)
            mir_prof_bool = (permuted_df["Profiles_similarity_score"] == 0)
            sim_prof_perm_num = len(permuted_df[sim_prof_bool])
            unsim_prof_perm_num = len(permuted_df[unsim_prof_bool])
            mir_prof_perm_num = len(permuted_df[mir_prof_bool])
            return {"similar": sim_prof_perm_num,
                    "unsimilar": unsim_prof_perm_num,
                    "mirror": mir_prof_perm_num,
                    "iteration": in_iter + 1,
                    "dataframe": permuted_df}
        permuted_df_results_temp = ptmp.ProcessingPool().map(f, range(e_value))
        self.prof_arr_perm_results = pd.DataFrame(permuted_df_results_temp)
        self.prof_arr_perm_res_avg = pd.Series({"mirror_profiles": sum(self.prof_arr_perm_results.mirror) /\
                                                                   len(self.prof_arr_perm_results),
                                                "similar_profiles": sum(self.prof_arr_perm_results.similar) /\
                                                                    len(self.prof_arr_perm_results),
                                                "unsimilar": sum(self.prof_arr_perm_results.unsimilar) /\
                                                             len(self.prof_arr_perm_results)})

    def KO_profs_perm(self,
                      e_value,
                      in_prof_sim_lev):
        def f(in_iter):
            q_ORF_prof_df = self.inter_df_stats[["Query_ORF",
                                                 "Query_gene_profile"]]
            a_ORF_prof_df = self.inter_df_stats[["Array_ORF",
                                                 "Array_gene_profile"]]
            drop_prof_temp_df = self.inter_df_stats.drop(["Query_gene_profile",
                                                          "Array_gene_profile",
                                                          "Profiles_similarity_score"],
                                                         axis = 1)
            q_ORF_prof_df.columns = range(len(q_ORF_prof_df.columns))
            a_ORF_prof_df.columns = range(len(a_ORF_prof_df.columns))
            stack_ORF_prof_df = pd.concat([q_ORF_prof_df,
                                           a_ORF_prof_df],
                                          ignore_index = True)
            stack_ORF_prof_df.drop_duplicates(inplace = True)
            stack_ORF_prof_df.columns = ["ORF", "Profile"]
            stack_ORF_prof_df.index = range(len(stack_ORF_prof_df))
            stack_prof_perm_df = stack_ORF_prof_df.Profile.sample(len(stack_ORF_prof_df))
            stack_prof_perm_df.index = range(len(stack_prof_perm_df))
            ORF_prof_perm_df = pd.concat([stack_ORF_prof_df.ORF,
                                          stack_prof_perm_df],
                                         axis = 1)
            q_merged_df = pd.merge(drop_prof_temp_df,
                                   ORF_prof_perm_df,
                                   left_on = "Query_ORF",
                                   right_on = "ORF",
                                   how = "left")
            qa_merged_df = pd.merge(q_merged_df,
                                    ORF_prof_perm_df,
                                    left_on = "Array_ORF",
                                    right_on = "ORF",
                                    how = "left",
                                    suffixes=("_query", "_array"))
            qa_merged_score_df = df_based_profiles_scorer(qa_merged_df,
                                                          prof_1_col_name = "Profile_query",
                                                          prof_2_col_name = "Profile_array",
                                                          score_col_name = "Profiles_similarity_score")
            sim_prof_bool = (qa_merged_score_df["Profiles_similarity_score"] >=\
                             in_prof_sim_lev)
            unsim_prof_bool = (qa_merged_score_df["Profiles_similarity_score"] <\
                               in_prof_sim_lev) &\
                              (qa_merged_score_df["Profiles_similarity_score"] > 0)
            mir_prof_bool = (qa_merged_score_df["Profiles_similarity_score"] == 0)
            sim_prof_perm_num = len(qa_merged_score_df[sim_prof_bool])
            unsim_prof_perm_num = len(qa_merged_score_df[unsim_prof_bool])
            mir_prof_perm_num = len(qa_merged_score_df[mir_prof_bool])
            return {"similar": sim_prof_perm_num,
                    "unsimilar": unsim_prof_perm_num,
                    "mirror": mir_prof_perm_num,
                    "iteration": in_iter + 1,
                    "dataframe": qa_merged_score_df}
        permuted_df_results_temp = ptmp.ProcessingPool().map(f, range(e_value))
        self.prof_KO_perm_results = pd.DataFrame(permuted_df_results_temp)
        self.prof_KO_perm_res_avg = pd.Series({"mirror_profiles": sum(self.prof_KO_perm_results.mirror) /\
                                                                   len(self.prof_KO_perm_results),
                                                "similar_profiles": sum(self.prof_KO_perm_results.similar) /\
                                                                    len(self.prof_KO_perm_results),
                                                "unsimilar": sum(self.prof_KO_perm_results.unsimilar) /\
                                                             len(self.prof_KO_perm_results)})


    def e_val_calc(self):
        """Return Ortho_Stats.e_value (int) which is an expected number of
        interactions with positive DMF and similar gene profiles by chance.
        """
        self.e_value = (self.DMF_positive_num * self.sim_prof_num) /\
                        self.tot_inter_num

class KEGG_API:
    """Provides connectivity with the KEGG database. Functions ending with <tbl>
    download files provided by KEGG but DO NOT modify them. Modifications
    needed for data processing are made on pandas.DataFrame.
    """
    def __init__(self):
        self.home = "http://rest.kegg.jp"
        self.operations = {"db_statistics": "info",
                           "list_entry_ids": "list",
                           "find_by_keyword": "find",
                           "get_by_entry_no": "get",
                           "conv_2_outside_ids": "conv",
                           "find_X_ref": "link"}
        self.databases = {"pathway": "path",
                          "brite": "br",
                          "module": "md",
                          "orthology": "ko",
                          "genome": "genome",
                          "genomes": "gn",
                          "ligand": "ligand",
                          "compound": "cpd",
                          "glycan": "gl",
                          "reaction": "rn",
                          "rpair": "rp",
                          "rclass": "rc",
                          "enzyme": "ec",
                          "disease": "ds",
                          "drug": "dr",
                          "dgroup": "dg",
                          "environ": "ev"}
        self.organisms_ids_df = None
        self.id_conversions = {"ncbi_gene": "ncbi-geneid",
                               "ncbi_prot": "ncbi-proteinid",
                               "uniprot": "uniprot",
                               "kegg_id": "genes"}
        self.id_conversions_df = None
        self.org_db_X_ref_df = None
        self.query_ids_not_found = []

    def get_organisms_ids(self,
                          out_file_name,
                          skip_dwnld = False):
        """Get KEGG's organisms' IDs, genomes IDs and definitions. Data are
        downloaded to a local file and then made into pandas.DataFrame. File
        can be reused. Necessary for KEGG_API.org_name_2_kegg_id.

        Args:
            out_file_name (str): name for file to be downloaded
            skip_dwnld (bool): read existing file when <True>. Default <False>
        """
        if skip_dwnld == True:
            pass
        else:
            url = "{0}/{1}/{2}".format(self.home,
                                       self.operations["list_entry_ids"],
                                       self.databases["genome"])
            res = rq.get(url)
            with open(out_file_name, "w") as fout:
                fout.write(res.content)
        self.organisms_ids_df = pd.read_csv(out_file_name,
                                     names = ["genome_id",
                                              "names",
                                              "description"],
                                     header=None,
                                     sep = "\t|;",
                                     engine = "python")
        temp_sub_df = self.organisms_ids_df["names"].str.split(",", expand = True)
        temp_sub_df.columns = ["kegg_org_id", "name", "taxon_id"]
        self.organisms_ids_df.drop("names", axis = 1, inplace = True)
        self.organisms_ids_df = pd.concat([self.organisms_ids_df, temp_sub_df], axis=1)
        self.organisms_ids_df.replace({"genome:":""},
                               regex=True,
                               inplace=True)
        self.organisms_ids_df.dropna(inplace=True)

    def org_name_2_kegg_id(self,
                           organism,
                           assume_1st = True):
        """Return KEGG's organisms' IDs (str) when queried  with a regular
        (natural) biological name. Case-sensitive. Uses KEGG_API.organisms_ids_df
        generated by KEGG_API.get_organisms_ids. Necessary for creation of ids
        list which is then passed to Genome.KO_list_profiler.

        Args:
            organism (str): biological organism's name to query against
            the KEGG's IDs
            assume_1st (bool): return the first item if more than one hit when
            <True> (default)
        """
        org_bool = self.organisms_ids_df.description.str.contains(organism)
        organism_ser = self.organisms_ids_df[org_bool]
        if len(organism_ser) == 0:
            print "No record found for {}".format(organism)
            self.query_ids_not_found.append(organism)
        elif len(organism_ser) > 1:
            if assume_1st == True:
                return organism_ser.kegg_org_id.iloc[0]
            print "More than one record for this query\n{}".format(organism_ser[["description",
                                                                                 "kegg_org_id"]])
        else:
            return str(organism_ser.kegg_org_id.to_string(index = False,
                                                          header = False))

    def get_id_conv_tbl(self,
                        source_id_type,
                        organism,
                        out_file_name,
                        skip_dwnld = False,
                        strip_pref = True):
        """Get genes or proteins IDs to KEGG IDs convertion table in
        pandas.DataFrame format. Data are downloaded to a local file and then
        made into pandas.DataFrame. File can be reused.

        Args:
            source_id_type (str): determines type of the source IDs
            organism (str): determines name of the organism bounded to the
            source IDs
            out_file_name (str): name for file to be downloaded
            skip_dwnld (bool) = read existing file when <True>. Default <False>
        """
        org_id = self.org_name_2_kegg_id(organism)
        if skip_dwnld == True:
            pass
        else:
            url = "{0}/{1}/{2}/{3}".format(self.home,
                                           self.operations["conv_2_outside_ids"],
                                           self.id_conversions[source_id_type],
                                           org_id)
            res = rq.get(url)
            with open(out_file_name, "w") as fout:
                fout.write(res.content)
        self.id_conversions_df = pd.read_csv(out_file_name,
                                              names = [source_id_type,
                                                       "kegg_id"],
                                              header = None,
                                              sep = "\t")
        if strip_pref == True:
            self.id_conversions_df.replace({"{0}:".format(org_id): ""},
                                           regex=True,
                                           inplace=True)
            self.id_conversions_df.replace({"{0}:".format(self.id_conversions[source_id_type]): ""},
                                           regex=True,
                                           inplace=True)
        else:
            pass

    def get_org_db_X_ref(self,
                         organism,
                         target_db,
                         out_file_name,
                         skip_dwnld = False,
                         strip_pref = True):
        """Get desired KEGG's database entries linked with all the genes from
        given organism. Data are downloaded to a local file and then made into
        pandas.DataFrame. File can be reused. Necessary for
        KEGG_API.get_ortho_db_entries and Ortho_Interactions.KO_based_appender.

        Args:
            organism (str): organism name. Provide whitespace-separated full
            species name. Uses pandas.series.str.contains method.
            targed_db (str): dict key for KEGG_API.databases of desired
            database.
            out_file_name (str): name for file to be downloaded
            skip_dwnld (bool) = read existing file when <True>. Default <False>
        """
        org_id = self.org_name_2_kegg_id(organism)
        if skip_dwnld == True:
            pass
        else:
            url = "{0}/{1}/{2}/{3}".format(self.home,
                                           self.operations["find_X_ref"],
                                           self.databases[target_db],
                                           org_id)
            res = rq.get(url)
            with open(out_file_name, "w") as fout:
                fout.write(res.content)
        self.org_db_X_ref_df = pd.read_csv(out_file_name,
                                           names = ["ORF_id", "kegg_id"],
                                           header=None,
                                           sep = "\t")
        if strip_pref == True:
            self.org_db_X_ref_df.replace({"{0}:".format(org_id): ""},
                                         regex = True,
                                         inplace = True)
            self.org_db_X_ref_df.replace({"{0}:".format(self.databases["orthology"]): ""},
                                         regex=True,
                                         inplace=True)
        else:
            pass

    def get_ortho_db_entries(self,
                             out_file_name):
        """Get full information about ortho groups by entries from
        KEGG_API.org_ortho_db_X_ref_df and download them into a local file.
        Necessary for Genome.parse_KO_db. The only func that does NOT convert
        downloaded file into pandas.DataFrame. Uses KEGG_API.get_org_db_X_ref_df.

        Args:
            out_file_name (str): name for file to be downloaded
        """
        counter = 0
        entries = self.org_db_X_ref_df["kegg_id"]
        for i in entries:
            counter += 1
            sign_prog_counter(counter, entries)
            url = "{0}/{1}/{2}".format(self.home,
                                       self.operations["get_by_entry_no"],
                                       i)
            res = rq.get(url)
            with open(out_file_name, "a") as fout:
                fout.write(res.content)

class Costanzo_API:
    """Provides connectivity with the Costanzo's SOM website of the Genetic
    Landscape of the Cell project, allowing data files download.

    Attribs:
        home (str): Costanzo's SOM home page address
        raw (str): raw data link and file name
        raw_matrix (str): raw data genetic interactions matrix link and file
        name, Java Treeview format
        lenient_cutoff (str): p-value < 0.05 cutoff link and file name
        intermediate_cutoff (str): |genetic interaction score| > 0.08,
        p-value < 0.05 cutoff link and file name
        stringent_cutoff (str): genetic interaction score < -0.12,
        p-value < 0.05 or genetic interaction score > 0.16, p-value < 0.05 link
        and file name
        bioprocesses (str): bioprocesses annotations
        chemical_genomics (str): chemical genomics data
        query_list (str): query ORFs list
        array_list (str): array ORFs list
    """

    def __init__(self):
        self.home = "http://drygin.ccbr.utoronto.ca/~costanzo2009"
        self.raw = "sgadata_costanzo2009_rawdata_101120.txt.gz"
        self.raw_matrix = "sgadata_costanzo2009_rawdata_matrix_101120.txt.gz"
        self.lenient_cutoff = "sgadata_costanzo2009_lenientCutoff_101120.txt.gz"
        self.intermediate_cutoff = "sgadata_costanzo2009_intermediateCutoff_101120.txt.gz"
        self.stringent_cutoff = "sgadata_costanzo2009_stringentCutoff_101120.txt.gz"
        self.bioprocesses = "bioprocess_annotations_costanzo2009.xls"
        self.chemical_genomics = "chemgenomic_data_costanzo2009.xls"
        self.query_list = "sgadata_costanzo2009_query_list_101120.txt"
        self.array_list = "sgadata_costanzo2009_array_list.txt"

    def get_data(self,
                 data):
        """Get files from Costanzo's SOM website.

        Args:
            data (str): specifies the file to be downloaded.
            <raw> for raw dataset,
            <raw_matrix> for raw genetic interactions matrix,
            <lenient_cutoff> for lenient dataset,
            <intermediate_cutoff> for intermediate dataset,
            <stringent_cutoff> for stringent dataset,
            <bioprocesses> for bioprocesses dataset,
            <chemical_genomics> for chemical genomics dataset,
            <query_list> for list of query ORFs names,
            <array_list> for list of array ORFs names
            out_file_name (str): name for file to be downloaded. Automatically
            same as appropriate Costanzo_API attrib when set to <None>
        """
        if data == "raw":
            url = "{0}/{1}".format(self.home,
                                   self.raw)
            out_file_name = self.raw
        elif data == "raw_matrix":
            url = "{0}/{1}".format(self.home,
                                   self.raw_matrix)
            out_file_name = self.raw_matrix
        elif data == "lenient_cutoff":
            url = "{0}/{1}".format(self.home,
                                   self.lenient_cutoff)
            out_file_name = self.lenient_cutoff
        elif data == "intermediate_cutoff":
            url = "{0}/{1}".format(self.home,
                                   self.intermediate_cutoff)
            out_file_name = self.intermediate_cutoff
        elif data == "stringent_cutoff":
            url = "{0}/{1}".format(self.home,
                                   self.stringent_cutoff)
            out_file_name = self.stringent_cutoff
        elif data == "bioprocesses":
            url = "{0}/{1}".format(self.home,
                                   self.bioprocesses)
            out_file_name = self.bioprocesses
        elif data == "chemical_genomics":
            url = "{0}/{1}".format(self.home,
                                   self.chemical_genomics)
            out_file_name = self.chemical_genomics
        elif data == "query_list":
            url = "{0}/{1}".format(self.home,
                                   self.query_list)
            out_file_name = self.query_list
        elif data == "array_list":
            url = "{0}/{1}".format(self.home,
                                   self.array_list)
            out_file_name = self.array_list
        else:
            raise ValueError("unknown option for data arg")
        res = rq.get(url)
        with open(out_file_name, "w") as fout:
            fout.write(res.content)

class Ortho_Network:
    """Calculates and holds data about interactions in form of network,
    exportable to other software (e.g. Cytoscape) or drawable by matplotlib.

    Attribs:
        inter_df (pandas.DataFrame): DataFrame containing genetic interactions
        nwrk (networkx.Graph): network created upon Ortho_Network.inter_df
    """
    def __init__(self,
                 inter_df):
        self.inter_df = inter_df
        self.nwrk = None

    def create_nwrk(self,
                    col_1_name,
                    col_2_name):
        """Return Ortho_Network.nwrk upon pandas.DataFrame.

        Args:
            col_1_name (str): column name to take as nodes
            col_2_name (str): column name to take as nodes
        """
        self.nwrk = nx.from_pandas_dataframe(self.inter_df,
                                             col_1_name,
                                             col_2_name)

    def write_nwrk(self,
                   out_file_name,
                   out_file_format):
        """Write Ortho_Network.nwrk to file readable to other software.

        Args:
            out_file_name (str): file name to save as
            out_file_format (str): file format to save as
        """
        if out_file_format.lower() == "graphml":
            nx.write_graphml(self.nwrk, out_file_name)
        elif out_file_format.lower() == "gefx":
            nx.write_gexf(self.nwrk, out_file_name)
        elif out_file_format.lower() == "gml":
            nx.write_gml(self.nwrk, out_file_name)

    def draw_nwrk(self,
                  width = 20,
                  height = 20,
                  dpi = None,
                  node_size = 5,
                  save_2_file = False,
                  out_file_name = "network.png"):
        """Return matplotlib.pyplot.figure of Ortho_Network.nwrk and/or write it to
        <*.png> file.

        Args:
            width (int): figure width in inches. Set as speciefied in
            matplotlibrc file when <None>. Default: <20>
            height (int): figure height in inches. Set as speciefied in
            matplotlibrc file when <None>. Default: <20>
            dpi (int): figure resolution. Set as speciefied in
            matplotlibrc file when <None>. Default: <None>
            node_size (int): size of the nodes. Default: <5>
            save_2_file (bool): write to file when <True>. Default: <False>
            out_file_name (str): file name to save as

        """
        plt.figure(figsize = (width, height))
        nx.draw_networkx(self.nwrk,
                         node_size = node_size,
                         node_color = "r",
                         node_alpha = 0.4,
                         with_labels = False)
        if save_2_file == True:
            plt.savefig(out_file_name,
                        dpi = dpi)
        else:
            pass

class HTML_generator:

    def __init__(self,
                 template_file,
                 definitions_file):
        self.template_file = template_file
        self.definitions_file = definitions_file
        self.template = None
        self.template_rendered = None

    def load_definitions(self):
        with open(self.definitions_file, "r") as fin:
            self.definitions = fin.readlines()
        self.definitions = [i.rstrip() for i in self.definitions]

    def load_template(self):
        template_Loader = jj2.FileSystemLoader(searchpath = ".")
        template_Env = jj2.Environment(loader = template_Loader)
        self.template = template_Env.get_template(self.template_file)

    def render_template(self,
                        filters,
                        DMF_positive,
                        DMF_negative,
                        mirror_profiles,
                        mirror_profiles_perm,
                        similar_profiles,
                        similar_profiles_perm,
                        dissimilar_profiles,
                        dissimilar_profiles_perm,
                        total,
                        e_value,
                        histogram_profs,
                        histogram_gis,
                        bivar,
                        lin_regr,
                        dataframe):
        curr_time = time.localtime()
        time_stamp = "{0}.{1}.{2}, {3}:{4}:{5}".format(curr_time.tm_year,
                                                       curr_time.tm_mon,
                                                       curr_time.tm_mday,
                                                       curr_time.tm_hour,
                                                       curr_time.tm_min,
                                                       curr_time.tm_sec)
        template_Vars = {"time_stamp": time_stamp,
                         "filters": filters,
                         "definitions": self.definitions,
                         "DMF_positive": DMF_positive,
                         "DMF_negative": DMF_negative,
                         "mirror_profiles": mirror_profiles,
                         "mirror_profiles_perm": mirror_profiles_perm,
                         "similar_profiles": similar_profiles,
                         "similar_profiles_perm": similar_profiles_perm,
                         "dissimilar_profiles": dissimilar_profiles,
                         "dissimilar_profiles_perm": dissimilar_profiles_perm,
                         "total": total,
                         "e_value": e_value,
                         "histogram_profs": histogram_profs,
                         "histogram_gis": histogram_gis,
                         "bivar": bivar,
                         "lin_regr": lin_regr,
                         "dataframe": dataframe}
        self.template_rendered = self.template.render(template_Vars)

    def save_template(self,
                      out_file_name):
        with open("{0}.html".format(out_file_name), "w") as fout:
            fout.write(self.template_rendered)

    def print_template(self):
        print self.template.render()

def main():
    pass

if __name__ == "__main__":
    main()
