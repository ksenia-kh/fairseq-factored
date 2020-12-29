#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/finetune-baseline-sanders-hd-en-es.log

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/sanders-gender-debias/handcraft"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints7-es-ft-s-hd"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/"

source ~/.bashrc
conda activate myenv

mkdir -p $CP_DIR

stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/train.py $WORKING_DIR \
 --arch  transformer_wmt_en_de --share-all-embeddings  --optimizer adam \
 --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --scoring bleu --warmup-init-lr 1e-07 \
 --warmup-updates 4000 --lr 0.0001 --min-lr 1e-09 --dropout 0.1 --weight-decay 0.0001 \
 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4000 \
  --finetune-from-model $CP_DIR/checkpoint_last.pt --save-dir $CP_DIR --best-checkpoint-metric bleu --maximize-best-checkpoint-metric\
  --source-lang en_tokensS --target-lang es_tokensS --max-update 2000
