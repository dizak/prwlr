# -*- coding: utf-8 -*-


import math
import requests as rq
import numpy as np


def remove_from_list(element,
                     iterable):
    """
    Return list without given element from that list. Conversely to built-in
    methods it is fruitful function.

    Parameters
    -------
    element: object
        Element to be removed from the list.
    iterable: list, tuple, set
        Iterable from with the element should be removed.

    Returns
    -------
    list
        Cleaned up from the element.

    """
    if element in list(iterable):
        iterable.remove(element)
    return iterable


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
