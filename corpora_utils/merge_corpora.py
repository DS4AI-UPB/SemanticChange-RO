from sys import argv
import json
from os import path, mkdir

dir_name = argv[1]

map =  json.load(open(dir_name + '/map.json'))
merge = json.load(open(argv[2]))
year2lines = {}

for file_name in map['files']:
    with open(dir_name + '/' + file_name, 'r') as f:
        for line in f:
            if len(line.strip()) > 0:
                line = line.strip() + '\n'
                if map['files'][file_name] not in year2lines:
                    year2lines[map['files'][file_name]] = []
                year2lines[map['files'][file_name]].append(line)
merge_dir = merge['target_dir']

if not path.exists(merge_dir):
    mkdir(merge_dir)

for interval in merge['intervals']:
    start = merge['intervals'][interval]['start']
    end = merge['intervals'][interval]['end']
    with open(f"{merge_dir}/{map['name']}_{interval}.txt", 'w') as f:
        for year in year2lines:
            if year >= start and year <= end:
                for line in year2lines[year]:
                    f.write(line)