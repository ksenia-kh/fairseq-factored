#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/align_en_tokens_tags_fr_mt_new.log


FEATURE_TAGGER_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/preprocessing/stanford/feature_tagger_en_fr"
PYTHON="python"

# Activate conda environment
source ~/.bashrc
conda activate myenv

stdbuf -i0 -e0 -o0 $PYTHON $FEATURE_TAGGER_DIR/align_features_en_bpe_tokensS_mt.py
