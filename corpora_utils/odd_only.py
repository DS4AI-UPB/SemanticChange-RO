from sys import argv

with open(argv[1], 'r') as fin:
    lines = fin.readlines()

with open(argv[1], 'w') as fout:
    for index, line in enumerate(lines):
        if index%2 == 1:
            fout.write(line)