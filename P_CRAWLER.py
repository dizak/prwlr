#! /usr/bin/env python
# -*- coding: utf-8 -*-

import lxml.etree as et
import pandas as pd
import re
import math
import requests as rq
import numpy as np
import pathos.multiprocessing as ptmp
import networkx as nx
import matplotlib.pyplot as plt
import jinja2 as jj2
import time
import json
from networkx.readwrite import json_graph
from tqdm import tqdm

__author__ = "Dariusz Izak"


def all_possible_combinations_counter(subset_size,
                                      set_size):
    """
    Return a number (int) of all possible combinations of elements in size
    of a subset of a set.

    Parameters
    -------
    subset_size: int
        Size of the subset.
    set_size: int
        Size of the whole set.

    Returns
    -------
    int
        Number of all combinations.
    """
    f = math.factorial
    return f(set_size) / f(subset_size) / f(set_size - subset_size)


def gene_finder_by_attrib(in_attr,
                          in_val,
                          out_attr,
                          in_genes,
                          exact=True):
    """
    Return an atrribute (dict value) of a gene from the Genome.genes found
    by another attribute of choice.

    Parameters
    -------
    in_attr: str
        Attribute to search in.
    in_val: str
        Desired value of the attribute to look for.
    out_attr: str
        Attribute of which value is desired to be returned.
    in_genes: list of dicts
        Genome.genes to search against.
    exact: bool
        Whole phrase when <True>. One word in phrase when <False>.
        Default <True>

    Returns
    -------
    str
        An attribute of a Genome.genes.

    Examples
    -------
    >>> print gene_finder_by_attrib("prot_id", "P22139", "description",
                                    sc.genes)
    DNA-directed RNA polymerases I, II, and III subunit RPABC5
    """
    if exact is True:
        for i in in_genes:
            if i[in_attr] == in_val:
                return i[out_attr]
    else:
        for i in in_genes:
            if in_val in i[in_attr]:
                return i[out_attr]


def gene_profile_finder_by_name(in_gene_name,
                                in_all_gene_profiles,
                                conc=True):
    """
    Return a gene profile (tuple of str) starting from the second position
    within the tuple from Genome.gene_profiles found by its name.

    Parameters
    -------
    in_gene_name: str
        Desired name to look for.
    in_all_gene_profiles: list of tuples
        Genome.gene_profiles to search against.
    conc: bool
        Elements of profile merged when <True>. Unmodified when <False>.

    Returns
    -------
    tuple of str
        Profile sequence, ("+-++") or ("+", "-", "+", "+").

    Examples
    -------
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
            if conc is True:
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
                                 index=in_df.index,
                                 columns=[score_col_name])
    in_df = pd.concat([in_df,
                       temp_score_df],
                      axis=1)
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
                                                 conc=False)
    gene_profile_2 = gene_profile_finder_by_name(in_genes_pair[1],
                                                 in_all_gene_profiles,
                                                 conc=False)
    if isinstance(gene_profile_1, type(None)) or isinstance(gene_profile_2,
                                                            type(None)) is True:
        pass
    else:
        return simple_profiles_scorer(gene_profile_1, gene_profile_2)


class ParserError(Exception):
    """
    Inappropriate structure passed to parser.
    """
    pass


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
        KO_list (list of dicts): data from parsed KEGG ortho-database
        KO_df (pandas.DataFrame): KO_list converted to DataFrame
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
        for i in in_file_names:
            print "\nprocessing {0}".format(i)
            with open(i, "r") as xml_fin:
                tree = et.parse(xml_fin)
            root = tree.getroot()
            if self.ref_name == str(root[2].attrib["name"])[:-1]:
                query_species_tax_name = root[1].attrib["name"]
                query_species_orthologs = root[2][0][0]
            else:
                query_species_tax_name = root[2].attrib["name"]
                query_species_orthologs = root[1][0][0]
            self.query_species.append(str(query_species_tax_name)[:-1])
            for ii in tqdm(self.genes):
                for iii in query_species_orthologs[:]:
                    if ii["prot_id"] == iii.attrib["protId"]:
                        if "orthologs" in ii:
                            ii["orthologs"]["organism"].update({str(query_species_tax_name)[:-1]:
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
        if in_col_name not in self.orthologous_groups_df.columns:
            raise KeyError("{0} not in Genome.orthologous_groups_df.columns".
                           format(in_col_name))
        else:
            pass
        temp_dict_list_1 = []
        self.orthologous_groups_df = pd.read_csv(in_file_name,
                                                 sep="\t",
                                                 index_col=0)
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
        for i in tqdm(temp_dict_list_1):
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
        for i in tqdm(self.genes):
            for ii in self.orthologous_groups_dict:
                if i["prot_id"] in ii["prot_id"]:
                    org_temp_str_list = []
                    org_temp_utf_list = self.orthologous_groups_df.columns[self.orthologous_groups_df.ix[ii["ortho_group_number"]].notnull()]
                    for e in org_temp_utf_list:
                        org_temp_str_list.append(str(e).split(".")[0])
                    i["ortho_group"] = ii["ortho_group_number"]
                    i["orthologs"] = {"organism":
                                      {e:
                                       {"ortho_group_number": ii["ortho_group_number"]}
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
                 id_type="prot_id"):
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
                         remove_empty=True,
                         upperize_ids=True,
                         profile_list=False,
                         KO_list_2_df=True,
                         profiles_df=True,
                         remove_species_white_spaces=True,
                         deduplicate=True):
        """Return Genome.KO_list (list of dict) or Genome.KO_df (pandas.DataFrame)
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
            KO_list_2_df (bool): convert Genome.KO_list to pandas.DataFrame.
            Rows NOT containing profiles are removed and resulting
            pandas.DataFrame is reindexed as continuous int sequence!
            profiles_df (bool): append Genome.KO_df with sign-per-column
            profiles list
        """
        if remove_empty is True:
            species_ids = [i for i in species_ids if i is not None]
        if upperize_ids is True:
            species_ids = [i.upper() for i in species_ids]
        for i in self.KO_list:
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
            self.KO_df = pd.DataFrame(self.KO_list)
            self.KO_df = self.KO_df[-self.KO_df["profile"].isnull()]
            self.KO_df.index = range(len(self.KO_df))
            if deduplicate is True:
                self.KO_df = self.KO_df.drop_duplicates(subset=["entry"],
                                                        keep="first")
                self.KO_df.index = range(len(self.KO_df))
            else:
                pass
            if profiles_df is True:
                if remove_species_white_spaces is True:
                    profs_df = pd.DataFrame(self.KO_df.profile.map(lambda x:
                                                                   [i for i in x])
                                                              .tolist(),
                                            columns=[i.replace(" ", "_") for i in self.query_species])
                else:
                    profs_df = pd.DataFrame(self.KO_df.profile.map(lambda x:
                                                                   [i for i in x])
                                                              .tolist(),
                                            columns=self.query_species)
                self.KO_df = pd.concat([self.KO_df, profs_df], axis=1)
                self.KO_df.index = range(len(self.KO_df))
            else:
                pass
        else:
            pass


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


class Ortho_Stats:
    """Calculates and holds data about interactions array statistical
    properties.
    """
    def __init__(self,
                 query_specices,
                 gene_profiles,
                 inter_df):
        self.query_species = query_specices
        self.inter_df = inter_df
        self.gene_profiles = gene_profiles
        self.num_prop_res = None
        self.e_value = None
        self.perm_results = None
        self.prof_arr_perm_res_avg = None
        self.filters_used = "No filters"
        self.filters_name = "no_filters"

    def df_selector(self,
                    DMF=None,
                    SMF_below_one=True,
                    inter_score_min=None,
                    inter_score_max=None,
                    no_flat_plus=False,
                    no_flat_minus=False,
                    process=None,
                    profiles=None,
                    prof_sim_lev=None):
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
        if profiles is None and prof_sim_lev is not None:
            raise ValueError("No value for profiles")
        elif profiles is not None and prof_sim_lev is None:
            raise ValueError("No value for prof_sim_lev")
        else:
            pass
        self.filters_used = []
        self.filters_name = []
        positive_DMF_bool = ((self.inter_df["DMF"] >
                             self.inter_df["SMF_Q"]) &
                             (self.inter_df["DMF"] >
                              self.inter_df["SMF_A"]))
        negative_DMF_bool = ((self.inter_df["DMF"] <
                             self.inter_df["SMF_Q"]) &
                             (self.inter_df["DMF"] <
                              self.inter_df["SMF_A"]))
        SMF_below_one_bool = (self.inter_df["SMF_Q"] < 1.0) &\
                             (self.inter_df["SMF_A"] < 1.0)
        inter_score_max_bool = (self.inter_df["GIS"] < inter_score_max)
        inter_score_min_bool = (self.inter_df["GIS"] > inter_score_min)
        no_flat_plu_q_bool = (self.inter_df["PROF_Q"] !=
                              "+" * len(self.query_species))
        no_flat_min_q_bool = (self.inter_df["PROF_Q"] !=
                              "-" * len(self.query_species))
        no_flat_plu_a_bool = (self.inter_df["PROF_A"] !=
                              "+" * len(self.query_species))
        no_flat_min_a_bool = (self.inter_df["PROF_A"] !=
                              "-" * len(self.query_species))
        iden_proc_bool = (self.inter_df["BSS"] ==
                          "identical")
        diff_proc_bool = (self.inter_df["BSS"] ==
                          "different")
        if profiles is not None:
            sim_prof_bool = (self.inter_df["PSS"] >=
                             prof_sim_lev)
            unsim_prof_bool = (self.inter_df["PSS"] <
                               prof_sim_lev)
        else:
            pass
        if DMF == "positive":
            self.inter_df = self.inter_df[positive_DMF_bool]
            self.filters_used.append("DMF positive")
            self.filters_name.append("DMF_p")
        elif DMF == "negative":
            self.inter_df = self.inter_df[negative_DMF_bool]
            self.filters_used.append("DMF negative")
            self.filters_name.append("DMF_n")
        else:
            pass
        if SMF_below_one is True:
            self.inter_df = self.inter_df[SMF_below_one_bool]
            self.filters_used.append("SMF < 1.0")
            self.filters_name.append("SMF_blw_1")
        else:
            pass
        if isinstance(inter_score_max, float) is True:
            self.inter_df = self.inter_df[inter_score_max_bool]
            self.filters_used.append("Genetic interaction score < {0}".format(inter_score_max))
            self.filters_name.append("gis_{0}".format(inter_score_max))
        else:
            pass
        if isinstance(inter_score_min, float) is True:
            self.inter_df = self.inter_df[inter_score_min_bool]
            self.filters_used.append("Genetic interaction score > {0}".format(inter_score_min))
            self.filters_name.append("gis_{0}".format(inter_score_min))
        else:
            pass
        if no_flat_plus is True:
            self.inter_df = self.inter_df[no_flat_plu_q_bool]
            self.inter_df = self.inter_df[no_flat_plu_a_bool]
            self.filters_used.append("No plus-only (eg ++++++) profiles")
            self.filters_name.append("no_plus_flat")
        else:
            pass
        if no_flat_minus is True:
            self.inter_df = self.inter_df[no_flat_min_q_bool]
            self.inter_df = self.inter_df[no_flat_min_a_bool]
            self.filters_used.append("No minus-only (eg ------) profiles")
            self.filters_name.append("no_min_flat")
        else:
            pass
        if process == "identical":
            self.inter_df = self.inter_df[iden_proc_bool]
            self.filters_used.append("Identical bioprocesses")
            self.filters_name.append("iden_proc")
        elif process == "different":
            self.inter_df = self.inter_df[diff_proc_bool]
            self.filters_used.append("Different bioprocesses")
            self.filters_name.append("diff_proc")
        else:
            pass
        if profiles == "similar":
            self.inter_df = self.inter_df[sim_prof_bool]
            self.filters_used.append("Similar profiles")
            self.filters_name.append("sim_prof")
        elif profiles == "unsimilar":
            self.inter_df = self.inter_df[unsim_prof_bool]
            self.filters_used.append("Dissimilar profiles")
            self.filters_name.append("dis_prof")
        else:
            pass

    def df_num_prop(self,
                    in_prof_sim_lev=None):
        """Return Ortho_Stats.tot_inter_num (int),
        Ortho_Stats.DMF_positive_num (int),
        Ortho_Stats.DMF_negative_num (int),
        Ortho_Stats.sim_prof_num (int).

        Args:
            in_prof_sim_lev (int): defines minimal Genome.gene_profiles in
            Ortho_Stats.inter_df similarity treshold. Set to <None> to
            omit, eg when dispalying multiple thresholds at once.
            Default: <None>
        """
        if isinstance(self.filters_used, str) is True:
            self.filters_used = []
        else:
            pass
        if isinstance(self.filters_name, str) is True:
            self.filters_name = []
        else:
            pass
        positive_DMF_bool = ((self.inter_df["DMF"] >
                             self.inter_df["SMF_Q"]) &
                             (self.inter_df["DMF"] >
                              self.inter_df["SMF_A"]))
        negative_DMF_bool = ((self.inter_df["DMF"] <
                             self.inter_df["SMF_Q"]) &
                             (self.inter_df["DMF"] <
                              self.inter_df["SMF_A"]))
        sim_prof_bool = (self.inter_df["PSS"] >=
                         in_prof_sim_lev)
        unsim_prof_bool = (self.inter_df["PSS"] <
                           in_prof_sim_lev) &\
                          (self.inter_df["PSS"] > 0)
        mir_prof_bool = (self.inter_df["PSS"] == 0)
        if in_prof_sim_lev is None:
            self.num_prop_res = pd.Series({"total": len(self.inter_df),
                                           "DMF_positive": len(self.inter_df[positive_DMF_bool]),
                                           "DMF_negative": len(self.inter_df[negative_DMF_bool]),
                                           "histogram_bins": pd.value_counts(self.inter_df["PSS"])})
        else:
            self.num_prop_res = pd.Series({"total": len(self.inter_df),
                                           "DMF_positive": len(self.inter_df[positive_DMF_bool]),
                                           "DMF_negative": len(self.inter_df[negative_DMF_bool]),
                                           "similar_profiles": len(self.inter_df[sim_prof_bool]),
                                           "unsimilar_profiles": len(self.inter_df[unsim_prof_bool]),
                                           "mirror_profiles": len(self.inter_df[mir_prof_bool]),
                                           "histogram_bins": pd.value_counts(self.inter_df["PSS"])})
            self.filters_used.append("Profiles similarity threshold: {0}".format(in_prof_sim_lev))
            self.filters_name.append("prof_sim_th_{0}".format(in_prof_sim_lev))

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
            q_temp_df = self.inter_df[["GENE_Q", "PROF_Q"]]
            a_temp_df = self.inter_df[["GENE_A", "PROF_A"]]
            q_temp_perm_df = q_temp_df.sample(len(q_temp_df))
            a_temp_perm_df = a_temp_df.sample(len(a_temp_df))
            q_temp_perm_df.index = range(len(q_temp_perm_df))
            a_temp_perm_df.index = range(len(a_temp_perm_df))
            qa_temp_perm_df = pd.concat([q_temp_perm_df, a_temp_perm_df], axis=1)
            for ii in qa_temp_perm_df.itertuples():
                temp_score_list.append(df_qa_names_2_prof_score([getattr(ii, "GENE_Q"),
                                                                 getattr(ii, "GENE_A")],
                                                                self.gene_profiles))
            temp_score_df = pd.DataFrame(temp_score_list,
                                         index=qa_temp_perm_df.index,
                                         columns=["PSS"])
            qa_temp_perm_score_df = pd.concat([qa_temp_perm_df, temp_score_df],
                                              axis=1)
            sim_prof_bool = (qa_temp_perm_score_df["PSS"] >=
                             in_prof_sim_lev)
            unsim_prof_bool = (qa_temp_perm_score_df["PSS"] <
                               in_prof_sim_lev) &\
                              (qa_temp_perm_score_df["PSS"] > 0)
            mir_prof_bool = (qa_temp_perm_score_df["PSS"] == 0)
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
        q_sign_per_col_profs_cols = ["{0}_Q".format(i) for i in self.query_species]
        a_sign_per_col_profs_cols = ["{0}_A".format(i) for i in self.query_species]

        def f(in_iter):
            temp_score_list = []
            q_prof_temp_df = self.inter_df["PROF_Q"]
            a_prof_temp_df = self.inter_df["PROF_A"]
            drop_prof_temp_df = self.inter_df.drop(["PROF_Q",
                                                    "PROF_A",
                                                    "PSS"] +
                                                   q_sign_per_col_profs_cols +
                                                   a_sign_per_col_profs_cols,
                                                   axis=1)
            q_prof_perm_temp_df = q_prof_temp_df.sample(len(q_prof_temp_df))
            a_prof_perm_temp_df = a_prof_temp_df.sample(len(a_prof_temp_df))
            q_prof_perm_temp_df.index = drop_prof_temp_df.index
            a_prof_perm_temp_df.index = drop_prof_temp_df.index
            permuted_df = pd.concat([drop_prof_temp_df,
                                     q_prof_perm_temp_df,
                                     a_prof_perm_temp_df],
                                    axis=1)
            for ii in permuted_df.itertuples():
                temp_score_list.append([simple_profiles_scorer(np.array(list(getattr(ii, "PROF_Q"))),
                                                               np.array(list(getattr(ii, "PROF_A"))))])
            temp_score_df = pd.DataFrame(temp_score_list,
                                         index=permuted_df.index,
                                         columns=["PSS"])
            permuted_profs_df = pd.concat([permuted_df,
                                          temp_score_df],
                                          axis=1)
            sim_prof_bool = (permuted_profs_df["PSS"] >=
                             in_prof_sim_lev)
            unsim_prof_bool = (permuted_profs_df["PSS"] <
                               in_prof_sim_lev) &\
                              (permuted_profs_df["PSS"] > 0)
            mir_prof_bool = (permuted_profs_df["PSS"] == 0)
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
        """Return a new Ortho_Stats.inter_df which was stripped from
        gene_profiles data and then appended with gene_profiles again using
        a permuted gene_profiles list.

        Args:
            e_value (int): number of times to shuffle the pandas DataFrame
            in_prof_sim_lev(int): treshold for assuming profiles as similar or
            not
        """
        q_sign_per_col_profs_cols = ["{0}_Q".format(i) for i in self.query_species]
        a_sign_per_col_profs_cols = ["{0}_A".format(i) for i in self.query_species]
        drop_prof_temp_df = self.inter_df.drop(["PROF_Q",
                                                "PROF_A",
                                                "PSS"] +
                                               q_sign_per_col_profs_cols +
                                               a_sign_per_col_profs_cols,
                                               axis=1)

        def f(in_iter):
            gene_profs_perm_arr_list = []
            prof_score_temp_list = []
            qa_attrib_temp_list = []
            conc_qa_prof_temp_list = []
            gene_profs_names = [i[0] for i in self.gene_profiles]
            gene_profs_profs = [i[1:] for i in self.gene_profiles]
            gene_profs_names_ser = pd.Series(gene_profs_names)
            gene_profs_profs_ser = pd.Series(gene_profs_profs)
            gene_profs_names_ser_perm = gene_profs_names_ser.sample(len(gene_profs_names_ser))
            gene_profs_names_ser_perm.index = range(len(gene_profs_names_ser_perm))
            gene_profs_perm_df = pd.concat([gene_profs_names_ser_perm,
                                            gene_profs_profs_ser],
                                           axis=1)
            gene_profs_perm_df.columns = ["perm_names", "profiles"]
            for i in gene_profs_perm_df.itertuples():
                name_arr = np.array(getattr(i, "perm_names"))
                full_arr = np.append(name_arr, getattr(i, "profiles"))
                gene_profs_perm_arr_list.append(full_arr)
            for i in drop_prof_temp_df.itertuples():
                qa_attrib_temp_list.append([getattr(i, "GENE_Q"),
                                            getattr(i, "GENE_A")])
            for i in qa_attrib_temp_list:
                prof_score_temp_list.append(df_qa_names_2_prof_score(i,
                                                                     gene_profs_perm_arr_list))
                conc_qa_prof_temp_list.append([gene_profile_finder_by_name(i[0],
                                                                           gene_profs_perm_arr_list,
                                                                           conc=True),
                                               gene_profile_finder_by_name(i[1],
                                                                           gene_profs_perm_arr_list,
                                                                           conc=True)])
            prof_score_temp_df = pd.DataFrame(prof_score_temp_list,
                                              index=drop_prof_temp_df.index,
                                              columns=["PSS"])
            profs_pairs_temp_df = pd.DataFrame(conc_qa_prof_temp_list,
                                               index=drop_prof_temp_df.index,
                                               columns=["PROF_Q", "PROF_A"])
            permuted_df = pd.concat([drop_prof_temp_df,
                                     profs_pairs_temp_df,
                                     prof_score_temp_df],
                                    axis=1)
            sim_prof_bool = (permuted_df["PSS"] >=
                             in_prof_sim_lev)
            unsim_prof_bool = (permuted_df["PSS"] <
                               in_prof_sim_lev) &\
                              (permuted_df["PSS"] > 0)
            mir_prof_bool = (permuted_df["PSS"] == 0)
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
        self.prof_arr_perm_res_avg = pd.Series({"mirror_profiles": sum(self.prof_arr_perm_results.mirror) / len(self.prof_arr_perm_results),
                                                "similar_profiles": sum(self.prof_arr_perm_results.similar) / len(self.prof_arr_perm_results),
                                                "unsimilar": sum(self.prof_arr_perm_results.unsimilar) / len(self.prof_arr_perm_results)})

    def KO_profs_perm(self,
                      e_value,
                      in_prof_sim_lev):
        """Return Ortho_Stats.prof_arr_perm_results pandas.DataFrame containing
        number of similar, dissimilar, mirror profiles and complete permuted
        pandas.DataFrame itself. Return Ortho_Stats.prof_arr_perm_res_avg
        containing average numbers of similar, dissimilar and mirror profiles.
        The algorithm:
            1. Extract Ortho_Stats.inter_df["ORF", "PROF"].
            2. Strip the original DataFrame from these 2 cols.
            3. Make the non-redundant list.
            4. Shuffle PROF col using pandas.Series.sample method.
            5. Merge with the stripped DataFrame on ORF (how="left").
            6. Calculate the results.

        Args:
            e_value (int): number of times to shuffle the pandas DataFrame
            in_prof_sim_lev(int): treshold for assuming profiles as similar or
            not
        """
        def f(in_iter):
            q_ORF_prof_df = self.inter_df[["ORF_Q",
                                           "PROF_Q"]]
            a_ORF_prof_df = self.inter_df[["ORF_A",
                                           "PROF_A"]]
            drop_prof_temp_df = self.inter_df.drop(["PROF_Q",
                                                    "PROF_A",
                                                    "PSS"],
                                                   axis=1)
            q_ORF_prof_df.columns = range(len(q_ORF_prof_df.columns))
            a_ORF_prof_df.columns = range(len(a_ORF_prof_df.columns))
            stack_ORF_prof_df = pd.concat([q_ORF_prof_df,
                                           a_ORF_prof_df],
                                          ignore_index=True)
            stack_ORF_prof_df.drop_duplicates(inplace=True)
            stack_ORF_prof_df.columns = ["ORF", "PROF"]
            stack_ORF_prof_df.index = range(len(stack_ORF_prof_df))
            stack_prof_perm_df = stack_ORF_prof_df.PROF.sample(len(stack_ORF_prof_df))
            stack_prof_perm_df.index = range(len(stack_prof_perm_df))
            ORF_prof_perm_df = pd.concat([stack_ORF_prof_df.ORF,
                                          stack_prof_perm_df],
                                         axis=1)
            q_merged_df = pd.merge(drop_prof_temp_df,
                                   ORF_prof_perm_df,
                                   left_on="ORF_Q",
                                   right_on="ORF",
                                   how="left")
            qa_merged_df = pd.merge(q_merged_df,
                                    ORF_prof_perm_df,
                                    left_on="ORF_A",
                                    right_on="ORF",
                                    how="left",
                                    suffixes=("_Q", "_A"))
            qa_merged_score_df = df_based_profiles_scorer(qa_merged_df,
                                                          prof_1_col_name="PROF_Q",
                                                          prof_2_col_name="PROF_A",
                                                          score_col_name="PSS")
            sim_prof_bool = (qa_merged_score_df["PSS"] >=
                             in_prof_sim_lev)
            unsim_prof_bool = (qa_merged_score_df["PSS"] <
                               in_prof_sim_lev) &\
                              (qa_merged_score_df["PSS"] > 0)
            mir_prof_bool = (qa_merged_score_df["PSS"] == 0)
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
        self.prof_KO_perm_res_avg = pd.Series({"mirror_profiles": sum(self.prof_KO_perm_results.mirror) /
                                              len(self.prof_KO_perm_results),
                                              "similar_profiles": sum(self.prof_KO_perm_results.similar) /
                                               len(self.prof_KO_perm_results),
                                               "unsimilar": sum(self.prof_KO_perm_results.unsimilar) /
                                               len(self.prof_KO_perm_results)})

    def e_val_calc(self):
        """Return Ortho_Stats.e_value (int) which is an expected number of
        interactions with positive DMF and similar gene profiles by chance.
        """
        self.e_value = (self.DMF_positive_num * self.sim_prof_num) / self.tot_inter_num


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
                          skip_dwnld=False):
        """Get KEGG's organisms' IDs, genomes IDs and definitions. Data are
        downloaded to a local file and then made into pandas.DataFrame. File
        can be reused. Necessary for KEGG_API.org_name_2_kegg_id.

        Args:
            out_file_name (str): name for file to be downloaded
            skip_dwnld (bool): read existing file when <True>. Default <False>
        """
        if skip_dwnld is True:
            pass
        else:
            url = "{0}/{1}/{2}".format(self.home,
                                       self.operations["list_entry_ids"],
                                       self.databases["genome"])
            res = rq.get(url)
            with open(out_file_name, "w") as fout:
                fout.write(res.content)
        self.organisms_ids_df = pd.read_csv(out_file_name,
                                            names=["genome_id",
                                                   "names",
                                                   "description"],
                                            header=None,
                                            sep="\t|;",
                                            engine="python")
        temp_sub_df = self.organisms_ids_df["names"].str.split(",", expand=True)
        temp_sub_df.columns = ["kegg_org_id", "name", "taxon_id"]
        self.organisms_ids_df.drop("names", axis=1, inplace=True)
        self.organisms_ids_df = pd.concat([self.organisms_ids_df, temp_sub_df], axis=1)
        self.organisms_ids_df.replace({"genome:": ""},
                                      regex=True,
                                      inplace=True)
        self.organisms_ids_df.dropna(inplace=True)

    def org_name_2_kegg_id(self,
                           organism,
                           assume_1st=True):
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
            if assume_1st is True:
                return organism_ser.kegg_org_id.iloc[0]
            print "More than one record for this query\n{}".format(organism_ser[["description",
                                                                                 "kegg_org_id"]])
        else:
            return str(organism_ser.kegg_org_id.to_string(index=False,
                                                          header=False))

    def get_id_conv_tbl(self,
                        source_id_type,
                        organism,
                        out_file_name,
                        skip_dwnld=False,
                        strip_pref=True):
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
        if skip_dwnld is True:
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
                                             names=[source_id_type,
                                                    "kegg_id"],
                                             header=None,
                                             sep="\t")
        if strip_pref is True:
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
                         skip_dwnld=False,
                         strip_ORF_prefix=True,
                         strip_kegg_id_prefix=False):
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
        if skip_dwnld is True:
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
                                           names=["ORF_id", "kegg_id"],
                                           header=None,
                                           sep="\t")
        if strip_ORF_prefix is True:
            self.org_db_X_ref_df["ORF_id"] = self.org_db_X_ref_df["ORF_id"].replace({"{0}:".format(org_id): ""},
                                                                                    regex=True)
        else:
            pass
        if strip_kegg_id_prefix is True:
            self.org_db_X_ref_df["kegg_id"] = self.org_db_X_ref_df["kegg_id"].replace({"{0}:".format(self.databases[target_db]): ""},
                                                                                      regex=True)
        else:
            pass

    def get_db_entries(self,
                       out_file_name):
        """Get full database by quering entries from
        KEGG_API.org_db_X_ref_df and download them into a local file.
        Necessary for Genome.parse_KO_db. The only func that does NOT convert
        downloaded file into pandas.DataFrame. Uses KEGG_API.get_db_X_ref_df.

        Args:
            out_file_name (str): name for file to be downloaded
        """
        entries = self.org_db_X_ref_df["kegg_id"].drop_duplicates()
        for i in tqdm(entries):
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
        lenient_cutoff (str): GIS_P < 0.05 cutoff link and file name
        intermediate_cutoff (str): |genetic interaction score| > 0.08,
        GIS_P < 0.05 cutoff link and file name
        stringent_cutoff (str): genetic interaction score < -0.12,
        GIS_P < 0.05 or genetic interaction score > 0.16, GIS_P < 0.05 link
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
        self.sub_nwrk = None

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

    def get_subgrps(self):
        self.sub_nwrk = [i for i in nx.connected_component_subgraphs(self.nwrk)]

    def write_nwrk(self,
                   out_file_name,
                   out_file_format):
        """Write Ortho_Network.nwrk to file readable to other software.

        Args:
            out_file_name (str): file name to save as
            out_file_format (str): file format to save as. Available formats are:
            graphml, gefx, gml, json
        """
        if out_file_format.lower() == "graphml":
            nx.write_graphml(self.nwrk, out_file_name)
        elif out_file_format.lower() == "gefx":
            nx.write_gexf(self.nwrk, out_file_name)
        elif out_file_format.lower() == "gml":
            nx.write_gml(self.nwrk, out_file_name)
        elif out_file_format.lower() == "json":
            with open("{0}.{1}".format(out_file_name, out_file_format), "w") as fout:
                fout.write(json.dumps(json_graph.node_link_data(self.nwrk)))

    def draw_nwrk(self,
                  width=20,
                  height=20,
                  dpi=None,
                  node_size=5,
                  save_2_file=False,
                  out_file_name="network.png"):
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
        plt.figure(figsize=(width, height))
        nx.draw_networkx(self.nwrk,
                         node_size=node_size,
                         node_color="r",
                         node_alpha=0.4,
                         with_labels=False)
        if save_2_file is True:
            plt.savefig(out_file_name,
                        dpi=dpi)
        else:
            pass


class HTML_Generator:
    """Saves results e.g. tables, figures, charts as an elegant html, using
    jinja2.

    Attribs:
        template_file (html with jinja2 vars): html template for embedding
        elements like charts or tables
        definitions_file (text file): stores lines showed in the final html
        under <Definitions> h1 tag. Each denifition is separated by newline.
        template (jinja2.enviroment.Template): loaded from
        HTML_Generator.template_file, containing vars vals passed into it
        template_rendered (unicode str): template rendered into regular unicode,
        save-ready
    """
    def __init__(self,
                 template_file,
                 definitions_file):
        self.template_file = template_file
        self.definitions_file = definitions_file
        self.template = None
        self.template_rendered = None

    def load_definitions(self):
        """Return HTML_Generator.definitions from
        HTML_Generator.definitions_file.
        """
        with open(self.definitions_file, "r") as fin:
            self.definitions = fin.readlines()
        self.definitions = [i.rstrip() for i in self.definitions]

    def load_template(self):
        """Load jinja2.environment.Template from HTML_Generator.template_file.
        Search path relative.
        """
        template_Loader = jj2.FileSystemLoader(searchpath=".")
        template_Env = jj2.Environment(loader=template_Loader)
        self.template = template_Env.get_template(self.template_file)

    def render_template(self,
                        name=None,
                        filters_pos=None,
                        filters_neg=None,
                        filters_neu=None,
                        num_prop_res=None,
                        num_prop_perm=None,
                        histogram_bins=None,
                        e_value=None,
                        histogram_gis_pos=None,
                        bivar_pos=None,
                        lin_regr_pos=None,
                        histogram_gis_neg=None,
                        bivar_neg=None,
                        lin_regr_neg=None,
                        histogram_gis_neu=None,
                        bivar_neu=None,
                        lin_regr_neu=None,
                        dataframe_pos=None,
                        dataframe_neg=None,
                        dataframe_neu=None,
                        results_type="chart",
                        skip_perm_res=False):
        """Retrun HTML_Generator.rendered_template with vals from passed vars.
        Args:
            name (str): name for results, displayed in most upper h2 tag
            filters_pos (list): passed from Ortho_Stats.filters_used. Displayed
            in ul tag in Filters article, under h2 DMF positive tag.
            filters_neg (list): passed from Ortho_Stats.filters_used. Displayed
            in ul tag in Filters article, under h2 DMF negative tag.
            filters_neu (list): passed from Ortho_Stats.filters_used. Displayed
            in ul tag in Filters article, under h2 DMF positive tag.
            num_prop_res (pandas.Series): passed from Ortho_Stats.num_prop_res.
            Values of num_prop_res are displayed in cells of Summary if
            results_type == <tbl>
            num_prop_perm (pandas.Series): passed from
            Ortho_Stats.num_prop_perm. Values of num_prop_perm are displayed in
            cells of Summary if results_type == <tbl>
            histogram_bins (pandas.DataFrame): sorted histogram bins from
            Ortho_Stats.num_prop_res. Used drawChart (JavaScript) func in
            template_file IMPORTANT: created externally. Will be included in
            one the classes of this script.
            e_value (int): passed from Ortho_Stats.e_value. Number of
            Ortho_Stats.inter_df permutations.
            histogram_gis_pos (str): path to png file, displayed in Plots
            section, above DMF positive h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            bivar_pos (str): path to png file, displayed in Plots
            section, above DMF positive h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            lin_regr_pos (str): path to png file, displayed in Plots
            section, above DMF positive h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            histogram_gis_neg (str): path to png file, displayed in Plots
            section, above DMF negative h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            bivar_neg (str): path to png file, displayed in Plots
            section, above DMF negative h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            lin_regr_neg (str): path to png file, displayed in Plots
            section, above DMF negative h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            histogram_gis_neu (str): path to png file, displayed in Plots
            section, above DMF neutral h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            bivar_neu (str): path to png file, displayed in Plots
            section, above neutral h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            bivar_neu (str): path to png file, displayed in Plots
            section, above neutral h5 tag. IMPORTANT: created externally.
            Will be included in one the classes of this script.
            dataframe_pos (unicode): pandas.Dataframe converted to html table
            using pandas.DataFrame.to_html method. Displayed in
            <div id=dataframe_pos></div> if is not <None> (default).
            dataframe_neg (unicode): pandas.Dataframe converted to html table
            using pandas.DataFrame.to_html method. Displayed in
            <div id=dataframe_neg></div> if is not <None> (default).
            dataframe_neu (unicode): pandas.Dataframe converted to html table
            using pandas.DataFrame.to_html method. Displayed in
            <div id=dataframe_neu></div> if is not <None> (default).
        """
        curr_time = time.localtime()
        time_stamp = "{0}.{1}.{2}, {3}:{4}:{5}".format(curr_time.tm_year,
                                                       curr_time.tm_mon,
                                                       curr_time.tm_mday,
                                                       curr_time.tm_hour,
                                                       curr_time.tm_min,
                                                       curr_time.tm_sec)
        template_Vars = {"time_stamp": time_stamp,
                         "name": name,
                         "filters_pos": filters_pos,
                         "filters_neg": filters_neg,
                         "filters_neu": filters_neu,
                         "definitions": self.definitions,
                         "num_prop_res": num_prop_res,
                         "num_prop_perm": num_prop_perm,
                         "histogram_bins": histogram_bins,
                         "e_value": e_value,
                         "histogram_gis_pos": histogram_gis_pos,
                         "bivar_pos": bivar_pos,
                         "lin_regr_pos": lin_regr_pos,
                         "histogram_gis_neg": histogram_gis_neg,
                         "bivar_neg": bivar_neg,
                         "lin_regr_neg": lin_regr_neg,
                         "histogram_gis_neu": histogram_gis_neu,
                         "bivar_neu": bivar_neu,
                         "lin_regr_neu": lin_regr_neu,
                         "dataframe_pos": dataframe_pos,
                         "dataframe_neg": dataframe_neg,
                         "dataframe_neu": dataframe_neu,
                         "results_type": results_type,
                         "skip_perm_res": skip_perm_res}
        self.template_rendered = self.template.render(template_Vars)

    def save_template(self,
                      out_file_name):
        """Save rendered template to file.

        Args:
            out_file_name (str): name for file to be saved
        """
        with open("{0}.html".format(out_file_name), "w") as fout:
            fout.write(self.template_rendered)

    def print_template(self):
        """Print template as it is rendered to str(). Just for debugging. Will
        be removed.
        """
        print self.template.render()


def main():
    """Temporary blank.
    Will be replaced with proper argument parser in the future.
    """
    pass


if __name__ == "__main__":
    main()
