#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --mem=30G # Memory
#SBATCH --gres=gpu:1
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/mt_assign_synsets_all_en_de_attn.log

SCRIPT_PATH1="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/preprocessing/babelfy/assign_align_synsets_without_at_lemmas_mt.py"
SCRIPT_PATH2="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored/preprocessing/babelfy/assign_align_synsets_without_at_pos_mt.py"
PYTHON="python"

source ~/.bashrc
conda activate myenv

stdbuf -i0 -e0 -o0 $PYTHON $SCRIPT_PATH1

stdbuf -i0 -e0 -o0 $PYTHON $SCRIPT_PATH2
