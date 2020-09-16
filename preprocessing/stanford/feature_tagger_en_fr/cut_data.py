import os
import sys


PATH = '/home/ksenia/Documents/studies/MAI/Thesis/data/en-fr'
LANG = 'en'

def main():
	with open(os.path.join(PATH, 'corpus.tc' + '.' + LANG), 'r', encoding="utf8") as file:
		text = file.readlines()

	with open(os.path.join(PATH, 'corpus.tc.1000' + '.' + LANG), 'w', encoding="utf8") as file:
		file.writelines(text[:1000])

if __name__ == "__main__":
    #sys.settrace(main())
    main()