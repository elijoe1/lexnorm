#include "ngram/ngram.h"

#include <iostream>
#include <string>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    /*NGram ngrams2;
    ngrams2.loadBin(argv[1]);
    ngrams2.save(argv[2]);
    exit(1);*/
    if (argc < 3)
    {
        std::cout << "Usage: give <srcFile> <dstFile> \n";
        exit(1);
    }
    NGram ngrams;

    ngrams.loadBin(argv[1]);
    ngrams.save(argv[2]);
}

