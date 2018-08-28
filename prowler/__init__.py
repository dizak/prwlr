__author__ = "Dariusz Izak, IBB PAS"
__version__ = "0.0.0"

__all__ = ["apis",
           "databases",
           "errors",
           "network",
           "profiles",
           "stats",
           "utils"]


from prowler import *


def profilize_organism(
    organism,
    species,
    IDs=None,
    X_ref=None,
    KOs=None
):
    """
    Returns pandas.DataFrame with Phylogenetic Profile for each ORF name of an
    organism.

    Parameters
    -------
    organism: str
        Full biological name of the organism.
    species: list of str
        List of full biological names to build the Phylogenetic Profile.
    IDs: str, filename
        Filename of the KEGG Organism IDs. Downloaded to a temporary file if
        <None>.
    X_ref: str, filename
        Filename of the ORF-KEGG Orthology Group cross-reference.
        Downloaded to a temporary file if <None>.
    KOs: str, filename
        Filename of the KEGG Orthology Group-Organism cross-reference.
        Downloaded to a temporary file if <None>.

    Returns
    ------
    pandas.DataFrame
    """
    kegg_db = databases.KEGG('Orthology')
    kegg_db.parse_organism_info(
        organism=organism,
        reference_species=species,
        IDs=IDs,
        X_ref=X_ref,
        KOs=KOs,
    )
    return kegg_db.organism_info.drop(columns=databases.Columns.KEGG_ID)
