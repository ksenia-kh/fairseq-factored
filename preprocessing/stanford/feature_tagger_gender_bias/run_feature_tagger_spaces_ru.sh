#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH -x veuc09,veuc06
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/feature_tagger_ru_spaces_mt.log


FEATURE_TAGGER_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/"
PYTHON="python"

# Activate conda environment
source ~/.bashrc
conda activate myenvru

stdbuf -i0 -e0 -o0 $PYTHON $FEATURE_TAGGER_DIR/feature_tagger_spaces_ru.py