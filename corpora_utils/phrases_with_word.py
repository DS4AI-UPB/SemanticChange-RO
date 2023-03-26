from sys import argv
import json
from os import mkdir, path

original_corpus = open(argv[1] , 'r')
words = json.load(open(argv[2]))["tests"]
target_path = argv[3]

lines = original_corpus.readlines()
if not path.exists(target_path):
    mkdir(target_path)

for word in words:
    lines_with_word = []
    for line in lines:
        if word["word"] in line.split():
            lines_with_word.append(line)
    with open(target_path + '/' + word["word"] + '.txt', 'w') as f:
        for line in lines_with_word:
            f.write(line)