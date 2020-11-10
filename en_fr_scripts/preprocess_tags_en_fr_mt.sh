#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/preprocess-joined-bpe-tags.log


WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/de-en/"

TRN_PREF="corpus.tc"
VAL_PREF="dev"
TES_PREF="test"
PYTHON="python"

FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/"

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/de-en/de-en-preprocessed-bpe"

# Activate conda environment
source ~/.bashrc
conda activate myenv

SRC="en_deps"
TGT="de_tokensS"

#stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
#    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
#    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.fr_tokensS.txt --thresholdsrc 1

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

SRC="en_lemmas"
TGT="de_tokensS"

#stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
#    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
#    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.fr_tokensS.txt --thresholdsrc 1

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

SRC="en_pos"
TGT="de_tokensS"

#stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
#    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
#    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.fr_tokensS.txt --thresholdsrc 1

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

SRC="en_tags"
TGT="de_tokensS"

#stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
#    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
#    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.fr_tokensS.txt --thresholdsrc 1

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt
