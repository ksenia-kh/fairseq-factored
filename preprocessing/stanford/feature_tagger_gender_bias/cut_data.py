import os
import sys


#PATH = '/home/usuaris/veu/ksenia.kharitonova/tfm/data/europarl/en-fr/'
PATH = "/home/ksenia/Documents/studies/MAI/Thesis/data/en-fr"
LANG = 'fr'

def main():
	with open(os.path.join(PATH, 'corpus.tc' + '.' + LANG), 'r', encoding="utf8") as file:
		text = file.readlines()

	with open(os.path.join(PATH, 'corpus.tc.10000' + '.' + LANG), 'w', encoding="utf8") as file:
		file.writelines(text[:10000])

if __name__ == "__main__":
    #sys.settrace(main())
    main()
