
seed=$(($RANDOM));

FAIRSEQ_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/src/fairseq-factored"
ENCODINGS_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings/es"
VOCABULARY="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-es/en-es-preprocessed-bpe/dict.en_tokensS.txt"
LABELS="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en.txt"

# Activate conda environment
source ~/.bashrc
conda activate myenv

echo $seed

echo ''
echo 'Baseline En-Es'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-enes-b.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Es POS'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-enes-f-pos.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Es TAGS'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-enes-f-tags.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Es DEPS'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-enes-f-deps.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Es Lemmas'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-enes-f-lemmas.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Es Syn Lemmas'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-enes-f-syn-lemmas.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Es Syn Pos'
echo '************************************************'
echo ''
python $FAIRSEQ_DIR/gender_classification.py -e $ENCODINGS_DIR/encodings-enes-f-syn-pos.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed