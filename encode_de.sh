#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/ksenia.kharitonova/tfm/log/encode-de.log

WORKING_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/de-en/de-en-preprocessed-bpe"
CP="average_model.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored"
OUTPUT_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings/de"
DEST_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/de-en/"
N=3888
#mkdir tmp

# Activate conda environment
source ~/.bashrc
conda activate myenv


mkdir -p  $OUTPUT_DIR

#Bilingual de baseline
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints17-de-b-v"
OUTPUT="encodings-ende-b.json"
SRC="en_tokensS"
TGT="de_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
         --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe

# Factored de POS
CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints13-de-pos"
OUTPUT="encodings-ende-f-pos.json"
SRC1="en_tokensS"
SRC2="en_pos"
TGT="de_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False

# Factored de TAGS

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints14-de-tags"
OUTPUT="encodings-ende-f-tags.json"
SRC1="en_tokensS"
SRC2="en_tags"
TGT="de_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored de DEPS

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints15-de-deps"
OUTPUT="encodings-ende-f-deps.json"
SRC1="en_tokensS"
SRC2="en_deps"
TGT="de_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored de Lemmas

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints16-de-lemmas"
OUTPUT="encodings-ende-f-lemmas.json"
SRC1="en_tokensS"
SRC2="en_lemmas"
TGT="de_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored de Syn-Lemmas

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints26-de-syn"
OUTPUT="encodings-ende-f-syn-lemmas.json"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_lemmas"
TGT="de_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


# Factored de Syn-Pos

CP_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/log/checkpoints27-de-syn-pos"
OUTPUT="encodings-ende-f-syn-pos.json"
SRC1="en_tokensS"
SRC2="en_synsets_wo_at_pos"
TGT="de_tokensS"

cuda_visible_devices="" stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --lang-pairs ${SRC1}-${TGT},${SRC2}-${TGT} --source-lang ${SRC1} --target-lang ${TGT} --task factored_translation \
        --output-file $OUTPUT_DIR/$OUTPUT --remove-bpe --multiple-encoders False


