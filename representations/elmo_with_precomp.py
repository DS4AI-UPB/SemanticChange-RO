from representations.representation import Representation
from representations.distances import compute_distances_for_sets
import csv
import pandas as pd
import shutil

class ELMo(Representation):

    def __init__(self):
        self.word_clusters = [[], []]
        self.clusterings = [{}, {}]

    def distance_metrics(self):
        return ["pointwise_euclidean", "pointwise_cosine", 'pointwise_canberra', 'pointwise_jaccard', 'pointwise_manhattan', "jsd", "cluster_count"]

    def get_name(self):
        return "elmo"
    
    def load_corpus(self, path, index):
        words = {}
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row = list(row.values())
                word = row[0].strip()
                if word not in words:
                    words[word] = []
                vals = [float(x) for x in row[1:]]
                words[word].append(vals)

        for word in words:
            words[word] = pd.DataFrame(words[word])

        self.word_clusters[index] = words

    def load_data(self, path1 ,path2, _):
        self.load_corpus(path1, 0)
        self.load_corpus(path2, 1)
    
    def compare(self, word, distance_metric = None):
        words1 = list(self.word_clusters[0].keys())
        words2 = list(self.word_clusters[1].keys())
        if word not in words1 and word not in words2:
            return 0
        if word not in words1:
            return 1
        if word not in words2:
            return 1
        return compute_distances_for_sets(self.word_clusters[0][word], self.word_clusters[1][word], distance_metric)
    
    def save_extra_info(self, dir_name, path1, path2):
        shutil.copyfile(path1, f'{dir_name}/clusters_1.csv')
        shutil.copyfile(path2, f'{dir_name}/clusters_2.csv')
        return {"use_clusters": True}