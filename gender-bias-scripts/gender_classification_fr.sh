
seed=$(($RANDOM));

ENCODINGS_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings/fr"
VOCABULARY="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-preprocessed-bpe/dict.en_tokensS.txt"
LABELS="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en.txt"

# Activate conda environment
source ~/.bashrc
conda activate myenv

echo $seed

echo ''
echo 'Baseline En-Fr'
echo '************************************************'
echo ''
python gender_classification.py -e $ENCODINGS_DIR/encodings-enfr-b.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Fr POS'
echo '************************************************'
echo ''
python gender_classification.py -e $ENCODINGS_DIR/encodings-enfr-f-pos.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Fr TAGS'
echo '************************************************'
echo ''
python gender_classification.py -e $ENCODINGS_DIR/encodings-enfr-f-tags.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Fr DEPS'
echo '************************************************'
echo ''
python gender_classification.py -e $ENCODINGS_DIR/encodings-enfr-f-deps.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Fr Lemmas'
echo '************************************************'
echo ''
python gender_classification.py -e $ENCODINGS_DIR/encodings-enfr-f-lemmas.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Fr Syn Lemmas'
echo '************************************************'
echo ''
python gender_classification.py -e $ENCODINGS_DIR/encodings-enfr-f-syn-lemmas.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed

echo ''
echo 'Factored En-Fr Syn Pos'
echo '************************************************'
echo ''
python gender_classification.py -e $ENCODINGS_DIR/encodings-enfr-f-syn-pos.json \
    -v $VOCABULARY \
    -l $LABELS \
    -o 0 \
    -s $seed