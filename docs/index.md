[![Build Status](https://travis-ci.org/dizak/prwlr.svg?branch=master)](https://travis-ci.org/dizak/prwlr)

[![Downloads](https://pepy.tech/badge/prwlr)](https://pepy.tech/project/prwlr)

# Prwlr - profiles crawler

Prwlr integrates **Genetic Interactions** and **Phylogenetic Profiles**.

> Nothing is more fun that BLASTing each protein sequence from the organisms of interest!


Prwlr uses **[KEGG Orthology](http://www.genome.jp/kegg/ko.html)** to determine who is the ortholog of whom.
You don't have to download it manually - Prwlr uses its API!

> We all love to use 20-or-so different software pieces just to annotate one network! And to store the profiles in some unintelligible form!

**Phylogenetic Profiles** are simple python objects. They are represented as binary lists with characters of choice (but ```+``` and ```-``` are my favourite) and hold a couple of small-but-useful methods.

> Bioinformatics software is difficult to create so it should be hard for someone else!

Prwlr is numpy- and pandas-based wherever possible. It integrates well with [pandas.DataFrames](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html).

Let's use Prwlr!

Get the Phylogenetic Profiles for each of the organism's ORF.

```python
import prwlr as prwl

species=[
    'Aeropyrum pernix',
    'Agrobacterium fabrum',
    'Arabidopsis thaliana',
    'Bacillus subtilis',
    'Caenorhabditis elegans',
    'Chlamydophila felis',
    'Dictyostelium discoideum',
    'Drosophila melanogaster',
    'Escherichia coli',
    'Homo sapiens',
    'Plasmodium falciparum',
    'Staphylococcus aureus',
    'Sulfolobus islandicus',
    'Tetrahymena thermophila',
    'Trypanosoma cruzi',
    'Volvox carteri',
]

profiles = prwl.profilize_organism(
    organism="Saccharomyces cerevisiae",
    species=species
)

profiles.head()
```

|ORF_ID|PROF
|------|----
|YNL113W|-+--+-+-++--+++
|YNL130C|--------+------
|YNL141W|-++-+-++++---++
|YNL151C|-+--+---+------
|YNL162W|++--+-+-++-++++

Parse your **Genetic Interactions** network. It can come from the widely-known [Costanzo Network](http://science.sciencemag.org/content/353/6306/aaf1420) or from any other source.

```python
ExN_NxE = prwl.read_sga('./SGA_ExN_NxE.txt')
```

OK, now let's integrate it...

```python
ExN_NxE_profiles = prwl.merge_sga_profiles(
    ExN_NxE,
    profiles,
)
```

...and calculate the distances between the profiles!

```python
ExN_NxE_profiles_pss = prwl.calculate_pss(
    ExN_NxE_profiles,
    method='jaccard',
)
```

How does it look now?

```python
ExN_NxE_profiles_pss
```

|ORF_Q|GENE_Q|ENTRY_Q|PROF_Q|ORF_A|GENE_A|ENTRY_A|PROF_A|GIS|SMF_Q|SMF_A|DMF|PSS
|-----|------|-------|------|-----|------|-------|------|---|-----|-----|---|---
|YBL097W|brn1-16|K06676|-+----++-+----+-|YPL110C|gde1|K18696|------+--------+|0.0219|0.8542|1.0235|0.8962|0.8333333
|YBL097W|brn1-16|K06676|-+----++-+----+-|YPL115C|bem3|K19840|----------------|0.0121|0.8542|0.9865|0.8547|1.0
|YBL097W|brn1-16|K06676|-+----++-+----+-|YPL116W|hos3|K11484|----------------|-0.0147|0.8542|1.01|0.8481|1.0
|YBL097W|brn1-16|K06676|-+----++-+----+-|YPL119C|dbp1|K11594|-+--+-++-++--+-+|-0.0036|0.8542|1.013|0.8617|0.5555556
|YBL097W|brn1-16|K06676|-+----++-+----+-|YPL120W|vps30|K08334|-+--+-++-+----++|-0.0488|0.8542|0.871|0.6952|0.2857143
|YBL097W|brn1-16|K06676|-+----++-+----+-|YPL127C|hho1|K11275|-+--+-++-+-----+|0.0082|0.8542|0.996|0.8589|0.42857143
|YBL097W|brn1-16|K06676|-+----++-+----+-|YPL134C|odc1|K15110|----+-++-+-----+|-0.0139|0.8542|1.025|0.8616|0.5714286
|YBL097W|brn1-16|K06676|-+----++-+----+-|YPL135W|isu1|K22068|-+--+-++-++--++-|0.0368|0.8542|0.9295|0.8308|0.375
|YBL097W|brn1-16|K06676|-+----++-+----+-|YPL138C|spp1|K14960|-------+-+------|-0.0763|0.8542|0.9973|0.7756|0.6
|YBL097W|brn1-16|K06676|-+----++-+----+-|YPL140C|mkk2|K08294|----------------|-0.025|0.8542|1.011|0.8386|1.0

Maybe you would like to see what's inside on of the profiles?

```python
ExN_NxE_profiles_pss.iloc[0].PROF_A.get_present()
```

```
['DDI', 'VCN']
```

With something more human-readable?

```python
IDs_names = prwl.get_IDs_names(species)

[
    IDs_names[i]
    for i in ExN_NxE_profiles_pss.iloc[0].PROF_A.get_present()
]
```

```
['Dictyostelium discoideum', 'Volvox carteri']
```
