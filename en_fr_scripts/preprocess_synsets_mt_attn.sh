#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/preprocess-joined-bpe-synsets-all-mt-all-attn.log


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

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-fr/f-syn-lemmas"

SRC="en_synsets_wo_at_lemmas"
TGT="fr_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-fr/f-syn-pos"

SRC="en_synsets_wo_at_pos"
TGT="fr_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

# ES

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-es/en-es-preprocessed-bpe"

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-es/f-syn-lemmas"

SRC="en_synsets_wo_at_lemmas"
TGT="es_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-es/f-syn-pos"

SRC="en_synsets_wo_at_pos"
TGT="es_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

# DE

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/de-en/de-en-preprocessed-bpe"

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/de-en/f-syn-lemmas"

SRC="en_synsets_wo_at_lemmas"
TGT="de_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/de-en/f-syn-pos"

SRC="en_synsets_wo_at_pos"
TGT="de_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

# RU

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-ru/en-ru-preprocessed-bpe"

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-ru/f-syn-lemmas"

SRC="en_synsets_wo_at_lemmas"
TGT="ru_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-ru/f-syn-pos"

SRC="en_synsets_wo_at_pos"
TGT="ru_tokensS"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
       --testpref $WORKING_DIR/mt.bpe --destdir $WORKING_DIR --srcdict $DEST_DIR/dict.${SRC}.txt --tgtdict $DEST_DIR/dict.${TGT}.txt