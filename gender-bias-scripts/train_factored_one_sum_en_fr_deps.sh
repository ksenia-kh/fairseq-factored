#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:2
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/train-factored-wmt-one-sum-en-fr-deps-new.log

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-preprocessed-bpe"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints36-fr-deps-new"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/"

source ~/.bashrc
conda activate myenv

mkdir -p $CP_DIR

#stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/train.py $WORKING_DIR \
# --task factored_translation --arch factored_transformer_iwslt_de_en  --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 --min-lr 1e-09 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4000  --save-dir $CP_DIR --lang-pairs de-en,de_postags_at-en --max-update 50000 --factors-to-freeze de_postags_at-en --freeze-factors-epoch 7

stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/train.py $WORKING_DIR \
 --task factored_translation --arch factored_transformer_one_encoder_sum_wmt_en_fr_deps \
 --share-all-embeddings --optimizer adam --adam-betas '(0.9, 0.98)' \
 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
 --lr 0.0001 --min-lr 1e-09 --dropout 0.1 --weight-decay 0.0001 \
 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
 --max-tokens 4096  --save-dir $CP_DIR --lang-pairs en_tokensS-fr_tokensS,en_deps-fr_tokensS \
 --max-update 100000 --multiple-encoders False
