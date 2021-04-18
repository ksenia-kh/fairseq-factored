Linguistics4fairness:  Neutralizing GenderBias in Neural Machine Translation by Introducing Linguistic Knowledge
==========================================

This repository contains the code of my master thesis,
Linguistics4fairness:  Neutralizing GenderBias in Neural Machine 
Translation by Introducing Linguistic Knowledge. It is based on 
Factored Transformer architecture developed by Jordi Armengol-Estapé (the original repository is located [here](https://github.com/jordiae/fairseq-factored "here")). It is important to thoroughly read his Readme, Fairseq’s documentation[^1] and take a look at Fairseq's Github repository[^2] in order to understand the implementation.

The mt-gender dataset and corresponding software for analysing the resulting translations are used from the Gabriel Stanovsky's [repository](https://github.com/gabrielStanovsky/mt_gender "repository").

The encoder-decoder attention analysis based on vestor norms is implemented using Goro Kobayashi's [implementation](https://github.com/gorokoba560/norm-analysis-of-transformer "implementation") for a newer version of Fairseq and adapted for an older version. 

Overview
--------

The scripts are mostly intended for running in a high performance cluster with GPUs.

In addition to Jordi's implementation, all the relevant bash scripts for preprocessing, training and analysing data are located in the following directories:

- `preprocessing/babelfy`: Scripts for retrieving synsets from Babelfy,
assigning and aligning them.
- `preprocessing/stanford/feature_tagger_gender_bias`: Scripts for tagging and aligning classical linguistic features with Spacy.
- `gender-bias-scripts`: Bash scripts for fairseq preprocessing, training models, generating test and MT-gender translations as well as encoding the source embeddings and generating the encoder-decoder attention information.

Installation
-----

For running the factored architectures, the Python dependencies of
Fairseq must be satisfied (see `requirements.txt`).

    git clone https://github.com/ksenia-kh/fairseq-factored.git
	cd fairseq
    pip install --editable ./

[^1]: Fairseq documentation:
    <https://fairseq.readthedocs.io/en/latest/>.

[^2]: Fairseq repository: <https://github.com/pytorch/fairseq>.