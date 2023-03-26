from sys import argv
text_file = open(argv[1], 'r')
text = text_file.read()

#cleaning
text = text.lower()
words = text.split()
words = [word.strip('.,!;()[]') for word in words]
words = [word.replace("'s", '') for word in words]

#finding unique
unique = []
for word in words:
    if word not in unique:
        unique.append(word)

#sort
unique.sort()

f = open(argv[2], "w")

if argv[3] == 'elmo':
    f.write("</S>\n")
    f.write("<S>\n")
    f.write("@@UNKNOWN@@\n")

for word in unique:
    f.write(word+"\n")