
seed=$(($RANDOM));

ENCODINGS_DIR="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/encodings/fr"
VOCABULARY="/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/en-fr-preprocessed-bpe/dict.en_tokensS.txt"
LABELS="/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/en.txt"

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