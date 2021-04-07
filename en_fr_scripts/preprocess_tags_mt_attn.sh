#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/preprocess-joined-bpe-tags-all-mt.log


TRN_PREF="corpus.tc"
VAL_PREF="dev"
TES_PREF="test"
PYTHON="python"

FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/"


# Activate conda environment
source ~/.bashrc
conda activate myenv

# FR

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-preprocessed-bpe"

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-fr/f-deps"

SRC="en_deps"
TGT="fr_tokensS"

#stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
#    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
#    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.fr_tokensS.txt --thresholdsrc 1

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-fr/f-lemmas"

SRC="en_lemmas"
TGT="fr_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-fr/f-pos"

SRC="en_pos"
TGT="fr_tokensS"

#stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
#    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
#    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.fr_tokensS.txt --thresholdsrc 1

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-fr/f-tags"

SRC="en_tags"
TGT="fr_tokensS"

#stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
#    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
#    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.fr_tokensS.txt --thresholdsrc 1

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

# ES

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-es/en-es-preprocessed-bpe"

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-es/f-deps"

SRC="en_deps"
TGT="es_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-es/f-lemmas"

SRC="en_lemmas"
TGT="es_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-es/f-pos"

SRC="en_pos"
TGT="es_tokensS"

#stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
#    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
#    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.fr_tokensS.txt --thresholdsrc 1

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-es/f-tags"

SRC="en_tags"
TGT="es_tokensS"

#stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
#    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
#    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.fr_tokensS.txt --thresholdsrc 1

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

# DE

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/de-en/de-en-preprocessed-bpe"

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/de-en/f-deps"

SRC="en_deps"
TGT="de_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/de-en/f-lemmas"

SRC="en_lemmas"
TGT="de_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/de-en/f-pos"

SRC="en_pos"
TGT="de_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/de-en/f-tags"

SRC="en_tags"
TGT="de_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

# RU

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-ru/en-ru-preprocessed-bpe"

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-ru/f-deps"

SRC="en_deps"
TGT="ru_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-ru/f-lemmas"

SRC="en_lemmas"
TGT="ru_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-ru/f-pos"

SRC="en_pos"
TGT="ru_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-ru/f-tags"

SRC="en_tags"
TGT="ru_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt