#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/generate-mt-factored-one-sum-synsets-fr-new.log


WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-preprocessed-bpe"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_lemmas"
TGT="fr_tokensS"
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en-fr/"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints38-fr-syn"
CP="checkpoint_last.pt"
#CP="model.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored"

# Activate conda environment
source ~/.bashrc
conda activate myenv

#stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/generate.py $DEST_DIR --path $CP_DIR/$CP \
#	--beam 5 --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation --remove-bpe

stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/generate.py $DEST_DIR --path $CP_DIR/$CP \
	--beam 5 --batch-size 1 --lang-pairs $SRC1-$TGT,$SRC2-$TGT --task factored_translation --remove-bpe --target-lang fr_tokensS --multiple-encoders False
