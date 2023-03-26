from sys import argv
import random
import json

def count(freqs, word):
    if word in freqs:
        freqs[word] = freqs[word] + 1
    else:
        freqs[word] = 1

def n_random_other_words(n, freqs):
    count = 0
    replacers = []
    while count < n:
        random_word = random.choice(list(freqs.keys()))
        replacers.append(random_word)
        count += 1
    return replacers

def main():
    freqs = {}
    lines = []
    replacements = {}
    for line in open(f'corpora/{argv[1]}/{argv[2]}.txt', 'r'):
        line = line.strip().split()
        for word in line:
            count(freqs, word)
        lines.append(line)
    
    freqs = {k: v for k, v in sorted(freqs.items(), key=lambda item: item[1])}
    
    freqs_count = len(freqs)

    to_replace_count = int(argv[3])
    replaced_count = 0

    for i in range(int(freqs_count * 3/10) , int(freqs_count * 4/10)):
        word = list(freqs.keys())[i]
        replacements[word] = []
        freqs.pop(word)
        
        word_replacers = n_random_other_words(2, freqs)
        
        for replacer in word_replacers:
            if replacer in freqs:
                freqs.pop(replacer)
        
        replacements[word] += word_replacers
        replaced_count += 1
        if replaced_count == to_replace_count:
            break

    new_lines = []
    for line in lines:
        new_line = []
        for word in line:
            if word in replacements:
                new_line.append( random.choice(replacements[word]))
            else:
                new_line.append(word)
        new_lines.append(new_line)

    with open(f'corpora/{argv[1]}/{argv[2]}_mod{to_replace_count}.txt', 'w') as out:
        for line in new_lines:
            out.write(" ".join(line) + "\n")

    to_test = []
    for replacement_set in replacements.values():
        for word in replacement_set:
            to_test.append({
                "word": word,
                "expected": 1.0
            })

    test_dict = {
        "target": "demo/data",
        "language": "en",
        "name": f"synthetic_{argv[3]}",
        "description": f"synthetic corpus test with {argv[3]} words replaced",
        "corpora": [
            f'corpora/{argv[1]}/{argv[2]}.txt',
            f'corpora/{argv[1]}/{argv[2]}_mod{to_replace_count}.txt'
        ],
        "threshold" : 0.05,
        "tests": to_test
    }

    with open(f'tasks/{argv[2]}_mod_test{to_replace_count}.json', 'w') as out:
        json.dump(test_dict, out, indent=4)

main()