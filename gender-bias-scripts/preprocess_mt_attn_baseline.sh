#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/preprocess-joined-bpe-mt-all-attn.log

SRC="en_tokensS"
TGT="fr_tokensS"

TRN_PREF="corpus.tc"
VAL_PREF="dev"
TES_PREF="test"

PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/"

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-joined-bpe"
DEST_DIR2="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-preprocessed-bpe"

N_OP=32000


# Activate conda environment
source ~/.bashrc
conda activate myenv

# fr baseline

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-fr/baseline"

subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${SRC} --vocabulary-threshold 50 < ${WORKING_DIR}/mt.${SRC} > ${WORKING_DIR}/mt.bpe.${SRC}
subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${TGT} --vocabulary-threshold 50 < ${WORKING_DIR}/mt.${TGT} > ${WORKING_DIR}/mt.bpe.${TGT}

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR2/dict.${SRC}.txt \
       --tgtdict $DEST_DIR2/dict.${TGT}.txt --nwordstgt $N_OP --nwordssrc $N_OP
       #--trainpref $DEST_DIR/${TRN_PREF}.bpe  #--validpref $DEST_DIR/${VAL_PREF}.bpe \


# ES

SRC="en_tokensS"
TGT="es_tokensS"

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-es/en-es-joined-bpe"
DEST_DIR2="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-es/en-es-preprocessed-bpe"

# es baseline

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-es/baseline"

subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${SRC} --vocabulary-threshold 50 < ${WORKING_DIR}/mt.${SRC} > ${WORKING_DIR}/mt.bpe.${SRC}
subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${TGT} --vocabulary-threshold 50 < ${WORKING_DIR}/mt.${TGT} > ${WORKING_DIR}/mt.bpe.${TGT}

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR2/dict.${SRC}.txt \
       --tgtdict $DEST_DIR2/dict.${TGT}.txt --nwordstgt $N_OP --nwordssrc $N_OP
       #--trainpref $DEST_DIR/${TRN_PREF}.bpe  #--validpref $DEST_DIR/${VAL_PREF}.bpe \

# DE

SRC="en_tokensS"
TGT="de_tokensS"

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/de-en/de-en-joined-bpe"
DEST_DIR2="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/de-en/de-en-preprocessed-bpe"

# de baseline

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/de-en/baseline"

subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${SRC} --vocabulary-threshold 50 < ${WORKING_DIR}/mt.${SRC} > ${WORKING_DIR}/mt.bpe.${SRC}
subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${TGT} --vocabulary-threshold 50 < ${WORKING_DIR}/mt.${TGT} > ${WORKING_DIR}/mt.bpe.${TGT}

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR2/dict.${SRC}.txt \
       --tgtdict $DEST_DIR2/dict.${TGT}.txt --nwordstgt $N_OP --nwordssrc $N_OP
       #--trainpref $DEST_DIR/${TRN_PREF}.bpe  #--validpref $DEST_DIR/${VAL_PREF}.bpe \

# RU

SRC="en_tokensS"
TGT="ru_tokensS"

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-ru/en-ru-joined-bpe"
DEST_DIR2="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-ru/en-ru-preprocessed-bpe"

# ru baseline

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-ru/baseline"

subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${SRC} --vocabulary-threshold 50 < ${WORKING_DIR}/mt.${SRC} > ${WORKING_DIR}/mt.bpe.${SRC}
subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${TGT} --vocabulary-threshold 50 < ${WORKING_DIR}/mt.${TGT} > ${WORKING_DIR}/mt.bpe.${TGT}

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR2/dict.${SRC}.txt \
       --tgtdict $DEST_DIR2/dict.${TGT}.txt --nwordstgt $N_OP --nwordssrc $N_OP
       #--trainpref $DEST_DIR/${TRN_PREF}.bpe  #--validpref $DEST_DIR/${VAL_PREF}.bpe \
