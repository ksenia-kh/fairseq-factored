import os
import sys

import torch
from transformers import MarianTokenizer, MarianMTModel

model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

PATH = '/home/usuaris/veu/ksenia.kharitonova/tfm/data/mt/'
LANG = '>>fr<<'

def main():
	with open(os.path.join(PATH, 'en.txt'), 'r', encoding="utf8") as file:
		text = file.readlines()
	en_original = [line.split('\t')[2] for line in text]
	en_original_to_translate = [LANG+' '+line for line in en_original]
	translated = model.generate(**tokenizer.prepare_seq2seq_batch(en_original_to_translate))
	tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
	result = [en_original[i]+' ||| '+tgt_text[i]+'\n' for i in range(len(en_original))]
	with open(os.path.join(PATH, 'en-fr.txt'),'w', encoding="utf8") as f:
		for line in result:
			f.write(line)

if __name__ == "__main__":
    #sys.settrace(main())
    main()
