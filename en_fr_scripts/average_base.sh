#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/average17-2.log


WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/de-en/de-en-preprocessed-bpe"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints17-de-b-v"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/"

source ~/.bashrc
conda activate myenv


stdbuf -i0 -e0 -o0 $PYTHON  $FAIRSEQ_DIR/scripts/average_checkpoints.py --inputs $CP_DIR \
	--num-epoch-checkpoints 10 --output $CP_DIR/average_model.pt
