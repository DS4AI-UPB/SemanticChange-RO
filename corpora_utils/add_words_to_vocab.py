import json
import sys
from os import path
from nltk.tokenize import word_tokenize

file_list = json.load(open(sys.argv[1],'r'))

if path.isfile(sys.argv[2]):
    with open(sys.argv[2],'r') as vocab:
        existent_words = set(vocab.read().splitlines())
else:
    existent_words = set([])

for file in file_list:
    with open(file,'r') as f:
        for line in f:
            for word in word_tokenize(line):
                if word not in existent_words:
                    existent_words.add(word.lower())


with open(sys.argv[2],'w') as vocab:
    for word in existent_words:
        vocab.write(word+'\n')