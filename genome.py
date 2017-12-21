# -*- coding: utf-8 -*-


import lxml.etree as et
import pandas as pd
import re
import numpy as np
import pathos.multiprocessing as ptmp
from tqdm import tqdm


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
