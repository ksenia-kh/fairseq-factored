#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/generate-factored-wmt-one-sum-en-ru-synsets.log

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-ru/en-ru-preprocessed-bpe"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints40-ru-syn"
CP="average_model.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/"

source ~/.bashrc
conda activate myenv

#mkdir -p $CP_DIR

stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/generate.py $WORKING_DIR --path $CP_DIR/$CP \
	--beam 5 --batch-size 1 --lang-pairs en_tokensS-ru_tokensS,en_synsets_wo_at_lemmas-ru_tokensS --task factored_translation --remove-bpe --target-lang ru_tokensS --multiple-encoders False