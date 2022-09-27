# Chord Types and Ornamentation

This repository contains the code and data used for the paper
"Chord Types and Ornamentation".

## Data

### ABC+ (DCML)

Based on internal annotations from [the DCML harmony corpus](https://github.com/DCMLab/corpora)
(commit 7d4b155c46100e149de376a727d69c76f9042ece).
Parts of this corpus have already been published ([Beethoven String Quartets](https://github.com/DCMLab/ABC/), [Mozart Piano Sonatas](https://github.com/DCMLab/mozart_piano_sonatas/)),
the remaining subcorpora are work in progress and therefore not public.

The dataset has been preprocessed using `preprocess_dcml.py`.
A list of used pieces is recorded in `preprocess_dmcl.log`.
The processed chord data can be found in `data/dcml.tsv`.

### EWLD

Based on the Enhanced Wikifonia Leadsheet Dataset (Simonetta et al. 2018).
The input data have been obtained using
[the method documented by the EWLD authors](https://framagit.org/sapo/OpenEWLD).

The dataset has been processed using `preprocess_ewld.py`
and the resulting chord data can be found in `data/ewld.tsv`.
A list of used pieces is recorded in `preprocess_ewld.log`.

## Code and Results

The dependencies for the preprocessing and model code are listed in `requirements.txt`.
An example for how to set up a virtual environment
and installying a corresponding jupyter kernel
is given in `init.sh`.

The main model is implemented in `model_paper.ipynb`.
Its output is stored in `dcml_params.{json,pt}` and `ewld_params.{json,pt}`.

The cluster model is implemented in `model_paper_clustering.ipynb`.
Its output is stored in `cluster_experiments/results/params/`.
This notebook is intended to be run with GPU support
and therefore contains instructions for how to run it on Google Colab.

The plots shown in the paper are generated using `plots_paper.ipynb`.
