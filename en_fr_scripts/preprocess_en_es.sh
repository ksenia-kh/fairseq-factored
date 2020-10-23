#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/preprocess-joined-bpe.log

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-es/en-es-joined-bpe"
SRC="en_tokensS"
TGT="es_tokensS"

TRN_PREF="corpus.tc"
VAL_PREF="dev"
TES_PREF="test"

PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/"

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-es/en-es-joined-bpe"
DEST_DIR2="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-es/en-es-preprocessed-bpe"


N_OP=32000



# Activate conda environment
source ~/.bashrc
conda activate myenv

mkdir $DEST_DIR2

echo "apply joined bpe"

subword-nmt learn-joint-bpe-and-vocab --input ${WORKING_DIR}/${TRN_PREF}.${SRC} ${WORKING_DIR}/${TRN_PREF}.${TGT} -s $N_OP -o ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --write-vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${SRC} ${DEST_DIR}/${TRN_PREF}.vocab.${TGT}

subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${SRC} --vocabulary-threshold 50 < ${WORKING_DIR}/${TRN_PREF}.${SRC} > ${DEST_DIR}/${TRN_PREF}.bpe.${SRC}
subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${TGT} --vocabulary-threshold 50 < ${WORKING_DIR}/${TRN_PREF}.${TGT} > ${DEST_DIR}/${TRN_PREF}.bpe.${TGT}

subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${SRC} --vocabulary-threshold 50 < ${WORKING_DIR}/${VAL_PREF}.${SRC} > ${DEST_DIR}/${VAL_PREF}.bpe.${SRC}
subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${TGT} --vocabulary-threshold 50 < ${WORKING_DIR}/${VAL_PREF}.${TGT} > ${DEST_DIR}/${VAL_PREF}.bpe.${TGT}

subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${SRC} --vocabulary-threshold 50 < ${WORKING_DIR}/${TES_PREF}.${SRC} > ${DEST_DIR}/${TES_PREF}.bpe.${SRC}
subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${TGT} --vocabulary-threshold 50 < ${WORKING_DIR}/${TES_PREF}.${TGT} > ${DEST_DIR}/${TES_PREF}.bpe.${TGT}


stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $DEST_DIR/${TRN_PREF}.bpe --validpref $DEST_DIR/${VAL_PREF}.bpe --testpref $DEST_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR2  --nwordstgt $N_OP --nwordssrc $N_OP --joined-dictionary