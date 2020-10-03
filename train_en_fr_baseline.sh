#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/train-baseline-wmt-bpe-en-fr.log

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-preprocessed-bpe"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-preprocessed-bpe/checkpoints3"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/"

source ~/.bashrc
conda activate myenv

mkdir -p $CP_DIR

stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/train.py $WORKING_DIR \
 --arch  transformer_vaswani_wmt_en_fr_big  --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 --min-lr 1e-09 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4000  --save-dir $CP_DIR --source-lang en_tokensS --target-lang fr_tokensS --max-update 70000
