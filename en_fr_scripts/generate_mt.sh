#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/generate-mt-baseline.log


WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-preprocessed-bpe"
SRC="en_tokensS"
TGT="fr_tokensS"
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints2"
CP="checkpoint_last.pt"
#CP="model.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored"

# Activate conda environment
source ~/.bashrc
conda activate myenv

stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/generate.py $DEST_DIR --path $CP_DIR/$CP \
	--beam 5 --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation --remove-bpe
