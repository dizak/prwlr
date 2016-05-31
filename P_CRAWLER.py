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

    def profiler(self, id_type = "prot_id"):
        """Return Genome.gene_profiles (list of tuples) by iterating over
        Genome.query_species (tuple of str) Genome.ortho_genes (list of dicts)
        <orthologs><organism> values. +/- signs sequence resembles
        Genome.query_species order.
        """
        for i in self.ortho_genes:
            temp_arr = np.array([i[id_type]])
            for ii in self.query_species:
                if ii in i["orthologs"]["organism"]:
                    temp_arr = np.append(temp_arr, "+")
                else:
                    temp_arr = np.append(temp_arr, "-")
            self.gene_profiles.append(temp_arr)

class Ortho_Interactions:
    """Holds data about gene interactions array extracted from (csv) file.
    Merges these data with Genome.gene_profiles (list of tuples) and
    Genome.genes (list of dicts) selected values.

    Attribs:
        query_species_Genome (tuple of strs): passed from Genome
        genes_Genome (list of dicts): passed from Genome
        gene_profiles_Genome (list of tuples): passed from Genome
        interact_df (pandas.DataFrame): holds data about interactions from
        parsed csv file. Can be appended with
        Ortho_Interactions.df_profiles_and_score_appender
    """
    def __init__(self,
                 query_species_Genome,
                 genes_Genome,
                 gene_profiles_Genome):
        self.query_species_inter = query_species_Genome
        self.genes_inter = genes_Genome
        self.gene_profiles_inter = gene_profiles_Genome
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

    def df_profiles_and_score_appender(self,
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
        self.perm_results = None
        self.test_last_df_test = None

    def df_selector(self,
                    DMF = "positive",
                    no_flat = True,
                    process = "identical"):
        """Return pandas DataFrame selected to chosen DMF type (bool).

        Args:
            DMF (str): selects DMF type. Possible: <positive>, <negative> or
            omit DMF selection. Default: <positive>
            no_flat (bool): eliminates mirrors when selected. Default <True>
            process (str): selects bioprocesses similarity. Possible: <identical>,
            "different". Default: <identical>
        """
        positive_DMF_bool = (self.inter_df_stats["DMF"] >\
                             self.inter_df_stats["Query_SMF"]) &\
                            (self.inter_df_stats["DMF"] >\
                             self.inter_df_stats["Array_SMF"])
        negative_DMF_bool = (self.inter_df_stats["DMF"] <\
                             self.inter_df_stats["Query_SMF"]) &\
                            (self.inter_df_stats["DMF"] <\
                             self.inter_df_stats["Array_SMF"])
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
        if DMF == "positive":
            self.inter_df_stats = self.inter_df_stats[positive_DMF_bool]
        elif DMF == "negative":
            self.inter_df_stats = self.inter_df_stats[negative_DMF_bool]
        else:
            pass
        if no_flat == True:
            self.inter_df_stats = self.inter_df_stats[no_flat_plu_q_bool]
            self.inter_df_stats = self.inter_df_stats[no_flat_min_q_bool]
            self.inter_df_stats = self.inter_df_stats[no_flat_plu_a_bool]
            self.inter_df_stats = self.inter_df_stats[no_flat_min_a_bool]
        else:
            pass
        if process == "identical":
            self.inter_df_stats = self.inter_df_stats[iden_proc_bool]
        elif process == "different":
            self.inter_df_stats = self.inter_df_stats[diff_proc_bool]
        else:
            pass

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

    def prof_perm(self,
                  e_value,
                  in_prof_sim_lev):
        """Return pandas.DataFrame of number of different types of profiles scores,
        ech generated from pandas.DataFrame in which gene profiles were permuted but
        NOT the rest of the data. It is an equivalent of permuting parameters in
        the interactions network without changing the network's topology.
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
        self.tot_inter_num = len(self.inter_df_stats)
        self.DMF_positive_num = len(self.inter_df_stats[positive_DMF_bool])
        self.DMF_negative_num = len(self.inter_df_stats[negative_DMF_bool])
        self.sim_prof_num = len(self.inter_df_stats[sim_prof_bool])
        self.unsim_prof_num = len(self.inter_df_stats[unsim_prof_bool])
        self.mir_prof_num = len(self.inter_df_stats[mir_prof_bool])

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
        self.organisms_df = None
        self.id_conversions = {"ncbi_gene": "ncbi-geneid",
                               "ncbi_prot": "ncbi-proteinid",
                               "uniprot": "uniprot",
                               "kegg_id": "genes"}
        self.id_conversions_df = None
        self.get_org_db_X_ref_df = None

    def get_organisms_ids(self,
                          out_file_name,
                          skip_dwnld = False):
        """Get KEGG's organisms' IDs, genomes IDs and definitions. Data are
        downloaded to a local file and then made into pandas.DataFrame. File
        can be reused.

        Args:
            out_file_name (str): name for file to be downloaded
            skip_dwnld (bool) = read existing file when <True>. Default <False>
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
        self.organisms_df = pd.read_csv(out_file_name,
                                     names = ["genome_id",
                                              "names",
                                              "description"],
                                     header=None,
                                     sep = "\t|;",
                                     engine = "python")
        temp_sub_df = self.organisms_df["names"].str.split(",", expand = True)
        temp_sub_df.columns = ["kegg_org_id", "name", "taxon_id"]
        self.organisms_df.drop("names", axis = 1, inplace = True)
        self.organisms_df = pd.concat([self.organisms_df, temp_sub_df], axis=1)
        self.organisms_df.replace({"genome:":""},
                               regex=True,
                               inplace=True)
        self.organisms_df.dropna(inplace=True)

    def org_name_2_kegg_id(self,
                           organism):
        organism_ser = self.organisms_df[self.organisms_df.description.str.contains(organism)]
        org_id = str(organism_ser.kegg_org_id.to_string(index = False,
                                                        header = False))
        return org_id

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
        pandas.DataFrame. File can be reused.

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
        org_ortho_db_X_ref_df and download them into a local file.

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

def main():
    pass

if __name__ == "__main__":
    main()
