#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/preprocess-joined-bpe-tags-fr-new.log


WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-joined-bpe"

TRN_PREF="corpus.tc"
VAL_PREF="dev"
TES_PREF="test"
PYTHON="python"

FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/"

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-preprocessed-bpe"

# Activate conda environment
source ~/.bashrc
conda activate myenv

SRC="en_deps"
TGT="fr_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.fr_tokensS.txt --thresholdsrc 1

SRC="en_lemmas"
TGT="fr_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.fr_tokensS.txt --thresholdsrc 1

SRC="en_pos"
TGT="fr_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.fr_tokensS.txt --thresholdsrc 1

SRC="en_tags"
TGT="fr_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.fr_tokensS.txt --thresholdsrc 1
