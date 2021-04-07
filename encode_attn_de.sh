#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/encode-attn-de.log

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/de-en/de-en-preprocessed-bpe"
CP="average_model.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored"
OUTPUT_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings_attn/de"
N=100
#mkdir tmp

# Activate conda environment
source ~/.bashrc
conda activate myenv
git checkout attention-analysis


mkdir -p  $OUTPUT_DIR

: '
#Bilingual de baseline
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints17-de-b-v"
OUTPUT="extracted-attn-ende-b.pkl"
SRC="en_tokensS"
TGT="de_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
         --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe
'

# Factored de POS
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/de-en/f-pos"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints13-de-pos"
OUTPUT="extracted-attn-ende-f-pos.pkl"
SRC1="en_tokensS"
SRC2="en_pos"
TGT="de_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False

# Factored de TAGS

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/de-en/f-tags"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints14-de-tags"
OUTPUT="extracted-attn-ende-f-tags.pkl"
SRC1="en_tokensS"
SRC2="en_tags"
TGT="de_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored de DEPS

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/de-en/f-deps"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints15-de-deps"
OUTPUT="extracted-attn-ende-f-deps.pkl"
SRC1="en_tokensS"
SRC2="en_deps"
TGT="de_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored de Lemmas

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/de-en/f-lemmas"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints16-de-lemmas"
OUTPUT="extracted-attn-ende-f-lemmas.pkl"
SRC1="en_tokensS"
SRC2="en_lemmas"
TGT="de_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored de Syn-Lemmas

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/de-en/f-syn-lemmas"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints26-de-syn"
OUTPUT="extracted-attn-ende-f-syn-lemmas.pkl"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_lemmas"
TGT="de_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored de Syn-Pos

DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/attn-analysis/de-en/f-syn-pos"
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints27-de-syn-pos"
OUTPUT="extracted-attn-ende-f-syn-pos.pkl"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_pos"
TGT="de_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode_attention.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


