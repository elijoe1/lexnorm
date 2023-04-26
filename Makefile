.PHONY: data

language = british
size = 35
var = 1

all: data

data:
	python -m lexnorm.data.concatenate
	python -m lexnorm.data.lexicon
#	./data/cpp/load_ngrams ./data/external/monoise_data/twitter.ngr.bin ./data/interim/twitter_ngrams
#	./data/cpp/load_ngrams ./data/external/monoise_data/wiki.ngr.bin ./data/interim/wiki_ngrams
#	python -m lexnorm.data.word_ngrams
	# make test lexica
	cd data/external/scowl-2020.12.07; ./mk-list -v0 american 70 > ../../test/american-70.txt
	iconv -f iso-8859-1 -t utf-8 data/test/american-70.txt > data/test/american-70-utf.txt
	mv -f data/test/american-70-utf.txt data/test/american-70.txt
	cd data/external/scowl-2020.12.07; ./mk-list -v3 en_AU 95 > ../../test/australian-95.txt
	iconv -f iso-8859-1 -t utf-8 data/test/australian-95.txt > data/test/australian-95-utf.txt
	mv -f data/test/australian-95-utf.txt data/test/australian-95.txt
	cd data/external/scowl-2020.12.07; ./mk-list -v1 british 35 > ../../test/british-35.txt
	iconv -f iso-8859-1 -t utf-8 data/test/british-35.txt > data/test/british-35-utf.txt
	mv -f data/test/british-35-utf.txt data/test/british-35.txt