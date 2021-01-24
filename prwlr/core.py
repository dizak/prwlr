import gc as _gc
import pandas as _pd
import numpy as _np
from . import databases as _databases
from . import profiles as _profiles

class Columns(_databases.Columns):
    """
    Container for the columns names defined in this module.
    """
    SPLIT_SUF = '_SPLIT'
    REF = 'REF'
    QRY = 'QRY'
    REF_SPLIT = '{}{}'.format(REF, SPLIT_SUF)
    QRY_SPLIT = '{}{}'.format(QRY, SPLIT_SUF)
    PROF_Q = _databases.Columns.PROF_Q
    PROF_A = _databases.Columns.PROF_A
    STR_SEP = '|'

def get_IDs_names(
    species,
):
    """
    Returns dict of KEGG Organism IDs as keys and biological names as values.

    Parameters
    -------
    species: list of str
        List of full biological names to convert into KEGG Organism IDs.

    Returns
    ------
    dict
    """
    kegg_db = _databases.KEGG('Orthology')
    kegg_db.parse_organism_info(
        organism=None,
        reference_species=species,
        IDs=None,
        X_ref=None,
        KOs=None,
        IDs_only=True,
    )
    return {k.lower(): v for k, v in kegg_db.ID_name.items()}

def profilize_organism(*args, **kwargs):
    """
    Returns pandas.DataFrame with Phylogenetic Profile for each ORF name of an
    organism.

    Parameters
    -------
    organism: str
        Full biological name of the organism.
    reference_species: list of str
        List of full biological names to build the Phylogenetic Profile.
    IDs: str, path
        Filename of the KEGG Organism IDs. Downloaded to a temporary file if
        <None>.
    X_ref: str, path
        Filename of the ORF-KEGG Orthology Group cross-reference.
        Downloaded to a temporary file if <None>.
    KOs: str, path
        Filename of the KEGG Orthology Group-Organism cross-reference.
        Downloaded to a temporary file if <None>.
    threads: int
        Number of threads to utilize when downloading from KEGG. More means
        faster but can make KEGG block the download temporarily. Default: <2>

    Returns
    ------
    pandas.DataFrame
    """
    kegg_db = _databases.KEGG('Orthology')
    kegg_db.parse_organism_info(*args, **kwargs)
    return kegg_db.organism_info.drop(columns=_databases.Columns.KEGG_ID)

def read_sga(
    filename,
    version=2,
):
    """
    Returns pandas.DataFrame with Genetic Interaction Network from
    the Costanzo's SGA experiment either version 1 or 2.

    Parameters
    -------
    filename: str, path
        Filename of the SGA.
    version: int
        Version number of the Costanzo's SGA experiment. 1 or 2 available.

    Returns
    -------
    pandas.DataFrame
    """
    if version == 1:
        sga = _databases.SGA1()
    elif version == 2:
        sga = _databases.SGA2()
    else:
        raise errors.ParserError("Only versions 1 and 2 of Costanzo's SGA experiment are supported.")
    sga.parse(filename=filename)
    return sga.sga

def read_profiles(
    filename,
    **kwargs
):
    """
    Returns pandas.Series with prwlr.profiles.Profile objects from CSV file.
    Together with prwlr.core.save_profiles provides a convenient way of
    saving/reading-in prwlr.profiles.Profile objects to/from a flat text file.

    Parameters
    -------
    filename: str, path
        CSV file name.

    Returns
    ------
    pandas.Series
    """
    ref_qry_df = _pd.read_csv(filename, **kwargs)
    ref_qry_df[Columns.REF_SPLIT] = ref_qry_df[Columns.REF].str.split(Columns.STR_SEP)
    ref_qry_df[Columns.QRY_SPLIT] = ref_qry_df[Columns.QRY].str.split(Columns.STR_SEP)
    return ref_qry_df[[Columns.REF_SPLIT, Columns.QRY_SPLIT]].apply(
        lambda x: _profiles.Profile(
            reference=x[Columns.REF_SPLIT],
            query=x[Columns.QRY_SPLIT],
        ),
        axis=1,
    )

def save_profiles(
    series,
    filename,
    **kwargs
):
    """
    Writes pandas.Series with prwlr.profiles.Profile objects to CSV file.
    Together with prwlr.core.read_profiles provides a convenient way of
    saving/reading-in prwlr.profiles.Profile objects to/from a flat text file.

    Parameters
    -------
    Filename: str, path
        CSV file name.
    """
    _pd.DataFrame(
        {
            Columns.REF: series.apply(lambda x: x.reference).str.join(Columns.STR_SEP),
            Columns.QRY: series.apply(lambda x: x.query).str.join(Columns.STR_SEP),
        },
    ).to_csv(filename, **kwargs)

def read_network(
    filename,
    **kwargs
):
    """
    Returns pandas.DataFrame representing a Genetic Interaction Network with
    prwlr.profiles.Profile objects from CSV file. Together with
    prwlr.core.save_profiles provides a convenient way of saving/reading-in
    prwlr.profiles.Profile objects to/from a flat text file.

    Parameters
    -------
    filename: str, path
        CSV file name.

    Returns
    -------
    pandas.DataFrame
    """
    qry_ref_col = '{}_{}'.format(Columns.PROF_Q, Columns.REF)
    qry_qry_col = '{}_{}'.format(Columns.PROF_Q, Columns.QRY)
    arr_ref_col = '{}_{}'.format(Columns.PROF_A, Columns.REF)
    arr_qry_col = '{}_{}'.format(Columns.PROF_A, Columns.QRY)
    df = _pd.read_csv(filename, **kwargs)
    df[Columns.PROF_Q] = df[[qry_ref_col, qry_qry_col]].apply(
        lambda x: _profiles.Profile(
            reference=x[qry_ref_col].split(Columns.STR_SEP),
            query=x[qry_qry_col].split(Columns.STR_SEP),
        ),
        axis=1,
    )
    df[Columns.PROF_A] = df[[arr_ref_col, arr_qry_col]].apply(
        lambda x: _profiles.Profile(
            reference=x[arr_ref_col].split(Columns.STR_SEP),
            query=x[arr_qry_col].split(Columns.STR_SEP),
        ),
        axis=1,
    )
    return df.drop(columns=[
        qry_ref_col,
        qry_qry_col,
        arr_ref_col,
        arr_qry_col,
    ])

def save_network(
    dataframe,
    filename,
    **kwargs
):
    """
    Writes pandas.DataFrame representing a Genetic Interaction Network with
    prwlr.profiles.Profile objects to CSV file. Together with
    prwlr.core.save_profiles provides a convenient way of saving/reading-in
    prwlr.profiles.Profile objects to/from a flat text file.

    Parameters
    -------
    filename: str, path
        CSV file name.
    """
    qry_ref_col = '{}_{}'.format(Columns.PROF_Q, Columns.REF)
    qry_qry_col = '{}_{}'.format(Columns.PROF_Q, Columns.QRY)
    arr_ref_col = '{}_{}'.format(Columns.PROF_A, Columns.REF)
    arr_qry_col = '{}_{}'.format(Columns.PROF_A, Columns.QRY)
    df = dataframe.copy()
    df[qry_ref_col] = df[Columns.PROF_Q].apply(
        lambda x: x.reference,
    ).str.join(Columns.STR_SEP)
    df[qry_qry_col] = df[Columns.PROF_Q].apply(
        lambda x: x.query,
    ).str.join(Columns.STR_SEP)
    df[arr_ref_col] = df[Columns.PROF_A].apply(
        lambda x: x.reference,
    ).str.join(Columns.STR_SEP)
    df[arr_qry_col] = df[Columns.PROF_A].apply(
        lambda x: x.query,
    ).str.join(Columns.STR_SEP)
    df.drop(
        columns=[Columns.PROF_Q, Columns.PROF_A],
    ).to_csv(filename, **kwargs)
    del df; _gc.collect()

def merge_sga_profiles(
    sga,
    profiles
):
    """
    Returns Genetic Interaction Network from the Costanzo's SGA experiment with
    Phylogenetic Profiles.
    """
    merged = _pd.merge(
        left=sga,
        right=profiles,
        left_on=_databases.Columns.ORF_Q,
        right_on=_databases.Columns.ORF_ID,
        how="left",
    ).merge(
        right=profiles,
        left_on=_databases.Columns.ORF_A,
        right_on=_databases.Columns.ORF_ID,
        how="left",
        suffixes=(
            _databases.Columns.QUERY_SUF,
            _databases.Columns.ARRAY_SUF,
        )
    )
    merged.drop(
        columns=[
            _databases.Columns.ORF_ID_Q,
            _databases.Columns.ORF_ID_A,
        ],
        axis=1,
        inplace=True,
    )
    merged.dropna(inplace=True)
    merged.reset_index(
        drop=True,
        inplace=True,
    )
    return merged

def calculate_pss(
    network,
    method,
):
    """
    Returns Genetic Interaction Network from the Costanzo's SGA experiment with
    Profiles Similarity Score.

    Parameters
    -------
    network: pandas.DataFrame
    method: str

    Returns
    -------
    pandas.DataFrame
    """
    if method == 'pairwise':
        def pss(ar1, ar2):
            return sum(a == b for a, b in zip(ar1, ar2))
        pss_vect = _np.vectorize(pss)
        network[_databases.Columns.PSS] = pss_vect(
            network[_databases.Columns.PROF_Q].apply(lambda x: x.profile),
            network[_databases.Columns.PROF_A].apply(lambda x: x.profile),
            )
    else:
        network[_databases.Columns.PSS] = network.apply(
            lambda x: x[_databases.Columns.PROF_Q].calculate_pss(
                x[_databases.Columns.PROF_A],
                method=method,
                ),
            axis=1,
        )
    return network
