#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/interactive-generate-mt.log


WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-preprocessed-bpe"
SRC="en_tokensS"
TGT="fr_tokensS"
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-preprocessed-bpe/checkpoints"
CP="checkpoint_best.pt"
#CP="model.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/"

# Activate conda environment
source ~/.bashrc
conda activate myenv

#stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/generate.py $DEST_DIR --path $CP_DIR/$CP \
#    --beam 5 --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation --remove-bpe

stdbuf -i0 -e0 -o0 cat $DEST_DIR/mt.bpe.${SRC} | $PYTHON $FAIRSEQ_DIR/interactive.py $WORKING_DIR --path $CP_DIR/$CP \
	--beam 5 --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation --remove-bpe | tee $DEST_DIR/mt.${TGT}
