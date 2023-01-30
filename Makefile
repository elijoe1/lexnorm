.PHONY: data

language = british
size = 35
var = 1

all: data

data:
	python -m lexnorm.data.make_train
	python -m lexnorm.data.lexicon
#	cd data/external/scowl-2020.12.07; ./mk-list -v$(var) $(language) $(size) > ../../interim/$(language)-$(size).txt
#	iconv -f iso-8859-1 -t utf-8 data/interim/$(language)-$(size).txt > data/interim/$(language)-$(size)-utf.txt
#	mv -f data/interim/$(language)-$(size)-utf.txt data/interim/$(language)-$(size).txt