#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --mem=10G # Memory
#SBATCH --gres=gpu:1
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/hf_translate_mt_fr.log


FEATURE_TAGGER_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/preprocessing/stanford/feature_tagger_en_fr"
PYTHON="python"

# Activate conda environment
source ~/.bashrc
conda activate myenv

stdbuf -i0 -e0 -o0 $PYTHON $FEATURE_TAGGER_DIR/hf_translate_mt.py
