#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/train-baseline-wmt-bpe-en-es.log

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-es/en-es-preprocessed-bpe"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints7-es-b"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/"

source ~/.bashrc
conda activate myenv

mkdir -p $CP_DIR

stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/train.py $WORKING_DIR \
 --arch  transformer_wmt_en_de --share-all-embeddings  --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0001 --min-lr 1e-09 --dropout 0.1 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4000  --save-dir $CP_DIR --source-lang en_tokensS --target-lang es_tokensS --max-update 100000
