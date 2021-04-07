#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/encode-attn-ru.log

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-ru/en-ru-preprocessed-bpe"
CP="average_model.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored"
OUTPUT_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings_attn/ru"
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en-ru/"
N=100
#mkdir tmp

# Activate conda environment
source ~/.bashrc
conda activate myenv
git checkout attention-analysis

mkdir -p  $OUTPUT_DIR

: '
#Bilingual ru baseline
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints20-ru-b"
OUTPUT="extracted-attn-enru-b.pkl"
SRC="en_tokensS"
TGT="ru_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
         --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe
'

# Factored ru POS
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-ru/f-pos"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints21-ru-pos"
OUTPUT="extracted-attn-enru-f-pos.pkl"
SRC1="en_tokensS"
SRC2="en_pos"
TGT="ru_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False

# Factored ru TAGS

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-ru/f-tags"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints22-ru-tags"
OUTPUT="extracted-attn-enru-f-tags.pkl"
SRC1="en_tokensS"
SRC2="en_tags"
TGT="ru_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored ru DEPS

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-ru/f-deps"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints23-ru-deps"
OUTPUT="extracted-attn-enru-f-deps.pkl"
SRC1="en_tokensS"
SRC2="en_deps"
TGT="ru_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored ru Lemmas

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-ru/f-lemmas"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints24-ru-lemmas"
OUTPUT="extracted-attn-enru-f-lemmas.pkl"
SRC1="en_tokensS"
SRC2="en_lemmas"
TGT="ru_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored ru Syn-Lemmas

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-ru/f-syn-lemmas"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints40-ru-syn"
OUTPUT="extracted-attn-enru-f-syn-lemmas.pkl"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_lemmas"
TGT="ru_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored ru Syn-Pos

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/en-ru/f-syn-pos"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints41-ru-syn-pos"
OUTPUT="extracted-attn-enru-f-syn-pos.pkl"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_pos"
TGT="ru_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False