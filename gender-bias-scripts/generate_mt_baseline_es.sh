#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/generate-mt-baseline-es.log


WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-es/en-es-preprocessed-bpe"
SRC="en_tokensS"
TGT="es_tokensS"
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en-es/"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints7-es-b"
CP="checkpoint_last.pt"
#CP="model.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored"

# Activate conda environment
source ~/.bashrc
conda activate myenv

stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/generate.py $DEST_DIR --path $CP_DIR/$CP \
	--beam 5 --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation --remove-bpe
