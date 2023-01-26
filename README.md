# Chord Types and Ornamentation

This repository contains the code and data used for the paper
"Chord Types and Ornamentation".

Authors: Christoph Finkensiep, Petter Ericson, Sebastian Klassmann, Martin Rohrmeier.

## Data

### ABC+ (DCML)

Based on the DCML harmonic annotation standard.
Includes 
- [Beethoven, String Quartets](https://github.com/DCMLab/ABC/)
- [Mozart, Piano Sonatas](https://github.com/DCMLab/mozart_piano_sonatas/)
- [Corelli, Trio Sonatas](https://github.com/DCMLab/corelli/)
- the [Romantic Piano Corpus](https://github.com/DCMLab/romantic_piano_corpus)
  - Beethoven, Piano Sonatas
  - Chopin, Mazurkas
  - Debussy, Suite Bergamasque
  - Dvorak, Silhouettes
  - Grieg, Lyric Pieces
  - Liszt, Années de Pelerinage
  - Medtner, Tales
  - Schumann, Kinderszenen
  - Tchaikovsky, The Seasons

The exact version of the data is included in this repository as git submodules
under `data/dcml_corpora` and `data/romantic_piano_corpus`.
To check out the submodules, either clone the repository with `--recures-submodules`:
```shell
$ git clone --recurse-submodules https://github.com/DCMLab/probabilistic_harmony_model
```
or run the following command after cloning:
```shell
git submodule update --init --recursive
```

The DCML dataset can be preprocessed using `preprocess_dcml.py`.
The processed chord data can be found in `data/dcml.tsv`.
A list of used pieces has been recorded in `data/preprocess_dmcl.txt`.

### EWLD

The [Enhanced Wikifonia Leadsheet Dataset](https://zenodo.org/record/1476555) (Simonetta et al. 2018).
This corpus is not included in this repository for licensing reasons,
but can be requested from the authors for scientific purposes.
The preprocessing script expects the unpacked corpus under `data/ewld`.

The EWLD can be processed using `preprocess_ewld.py`
and the resulting chord data can be found in `data/ewld.tsv`.
A list of used pieces is recorded in `data/preprocess_ewld.txt`.

## Code and Results

The dependencies for the preprocessing and model code are listed in `requirements.txt`
(or `requirements-amd.txt` for AMD GPUs).
An example for how to set up a virtual environment
and installying a corresponding Jupyter kernel
is given in `init.sh`.
The default version will work with CPUs and Nvidia GPUs.
An alternative `pip install` command for AMD GPUs is included in `init.sh`.
The minimal library requirements without specified version are listed in `requirements-any.txt`.

The main model is implemented as a Jupyter notebook in `model.ipynb`.
Its output is stored in `results/dcml_params.{json,pt}` and `results/ewld_params.{json,pt}`.
The model can also be run as a normal Python script (without Jupyter) as `model.py`.
This will show all plots in separate windows at the end of the computation,
so running the notebook version is generally preferred.
The model can be run on CPU or GPU.

The cluster model is implemented in `model_clustering.ipynb` / `model_clustering.py`.
Its output is stored in `results/clustering/`.
This notebook is intended to be run with GPU support
and therefore contains instructions for how to run it on Google Colab.
Running it on CPU is possible but can take much longer.

The plots shown in the paper are generated using `plots.ipynb` / `plots.py`
and can be found in `plots/` directory.
Because of mathematical notation in some of the plots,
running this notebook requires a LaTeX installation.
Alternatively, LaTeX calls can be disabled at the top of the notebook.

Plots of the remaining posterior distributions not shown in the paper
can be found in [`additional-results.pdf`](additional-results.pdf).

## Licences

All code in this repository (i.e., all `.py` and `.ipynb` files, `init.sh`, and `requirements.txt`)
is licensed under the BSD 3-clause license (see `LICENSE-BSD`).
The licenses of the DCML datasets are provided in the respective subdirectories under `data/`.
All remaining documents, images, and data files are licensed under the CC-BY 4.0 license
(see `LICENSE-CCBY`).
