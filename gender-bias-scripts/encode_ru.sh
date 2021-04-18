#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/encode-ru.log

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-ru/en-ru-preprocessed-bpe"
CP="average_model.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored"
OUTPUT_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings/ru"
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en-ru/"
N=3888
#mkdir tmp

# Activate conda environment
source ~/.bashrc
conda activate myenv


mkdir -p  $OUTPUT_DIR

#Bilingual ru baseline
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints20-ru-b"
OUTPUT="encodings-enru-b.json"
SRC="en_tokensS"
TGT="ru_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
         --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe

# Factored ru POS
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints21-ru-pos"
OUTPUT="encodings-enru-f-pos.json"
SRC1="en_tokensS"
SRC2="en_pos"
TGT="ru_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False

# Factored ru TAGS

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints22-ru-tags"
OUTPUT="encodings-enru-f-tags.json"
SRC1="en_tokensS"
SRC2="en_tags"
TGT="ru_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored ru DEPS

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints23-ru-deps"
OUTPUT="encodings-enru-f-deps.json"
SRC1="en_tokensS"
SRC2="en_deps"
TGT="ru_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored ru Lemmas

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints24-ru-lemmas"
OUTPUT="encodings-enru-f-lemmas.json"
SRC1="en_tokensS"
SRC2="en_lemmas"
TGT="ru_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored ru Syn-Lemmas

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints40-ru-syn"
OUTPUT="encodings-enru-f-syn-lemmas.json"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_lemmas"
TGT="ru_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored ru Syn-Pos

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints41-ru-syn-pos"
OUTPUT="encodings-enru-f-syn-pos.json"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_pos"
TGT="ru_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


