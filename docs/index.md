[![Build Status](https://travis-ci.org/dizak/prowler.svg?branch=master)](https://travis-ci.org/dizak/prowler)

# Prowler - profiles crawler

Prowler integrates **Genetic Interactions** and **Phylogenetic Profiles**.

> Nothing is more fun that BLASTing each protein sequence from the organisms of interest!


Prowler uses **[KEGG Orthology](http://www.genome.jp/kegg/ko.html)** to determine who is the ortholog of whom.
You don't have to download it manually - Prowler uses its API!

> We all love to use 20-or-so different software pieces just to annotate one network! And to store the profiles in some unintelligible form!

**Phylogenetic Profiles** are simple python objects. They are represented as binary lists with characters of choice (but ```+``` and ```-``` are my favourite) and hold a couple of small-but-useful methods.

> Bioinformatics software is difficult to create so it should be hard for someone else!

Prowler is numpy- and pandas-based wherever possible. It integrates well with [pandas.DataFrames](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html).

Let's use Prowler!

  i. Parse [KEGG Orthology](http://www.genome.jp/kegg/ko.html).

  ```
  import prowler


  kegg_db = prowler.databases.KEGG("Orthology")

  kegg_db.parse_organism_info(organism="Saccharomyces cerevisiae",
                              reference_species=['Aeropyrum pernix',
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
                                                 'Volvox carteri'])

  kegg_db.parse_database("./KO_database.txt")
  ```
  ii. Parse your **Genetic Interactions** network. It can come from the widely-known [Costanzo Network](http://science.sciencemag.org/content/353/6306/aaf1420) or from any other source.

  ```
  sga2 = prowler.databases.SGA2()

  sga2.parse("./SGA_ExN_NxE.txt")
  ```

  iii. OK, now let's integrate it!

  ```
  profint = prowler.databases.ProfInt()

  profint.merger(kegg_db.database,
                 kegg_db.X_reference,
                 sga2.sga)

  profint.profilize(kegg_db.reference_species,
                    method="jaccard")
  ```

  How does it look now?

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
