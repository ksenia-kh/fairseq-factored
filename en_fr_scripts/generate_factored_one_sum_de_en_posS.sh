#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/generate-factored-wmt-one-sum-en-de-pos-160-avg5.log

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/de-en/de-en-preprocessed-bpe"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints13-de-pos"
CP="average_model.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/"

source ~/.bashrc
conda activate myenv

stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/generate.py $WORKING_DIR --path $CP_DIR/$CP \
	--beam 5 --batch-size 1 --lang-pairs en_tokensS-de_tokensS,en_pos-de_tokensS --task factored_translation --remove-bpe --target-lang de_tokensS --multiple-encoders False
