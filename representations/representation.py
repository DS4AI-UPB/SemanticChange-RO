import json
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

class Representation():
    """
    Representation is the base class for all representations.
    """
    
    def distance_metrics(self):
        return []

    def get_name(self):
        return ""

    def train(self):
        pass

    def load_model(self, path):
        pass

    def save_model(self, path):
        pass

    def load_data(self, path1 ,path2, targets):
        pass

    def get_embeddings(self, word):
        return []

    def compare(self, word, distance_metric = None):
        return 0

    def distance(self, word1, word2):
        return 0

    def save_extra_info(self, dir_name, path1, path2):
        pass

    def do_test(self, path):
        test_file = open(path)
        test_json = json.load(test_file)
        threshold = test_json["threshold"]

        self.load_data(test_json["corpora"][0], test_json["corpora"][1],[test["word"] for test in test_json["tests"]])
        
        if not "skip_training" in test_json:
            self.train()

        dict = {}
        dict['name'] = f"{self.get_name()}_{test_json['name']}"
        dict['language'] = test_json['language']
        dict['tesk_description'] = test_json['description']
        dict['distance_metrics'] = self.distance_metrics()
        dict['results'] = []
        dir_name = f'{test_json["target"]}/{self.get_name()}_{test_json["name"]}'

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        for distance_metric in self.distance_metrics():
            self.actuals = []
            self.expecteds = []
            self.words = []
            self.results = []
            self.distances = []
            self.detected = []

            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0


            for test in test_json['tests']:
                word = test['word']
                result = self.compare(word, distance_metric)
                self.results.append(result)
                if result is None:
                    continue
                self.distances.append(result)
            
            max_distance = np.max(self.distances)
            min_distance = np.min(self.distances)
            if max_distance - min_distance != 0:
                self.distances = [(distance - min_distance) / (max_distance - min_distance) for distance in self.distances]
            for distance, result, test in zip(self.distances,self.results, test_json['tests']):
                word = test['word']
                if result is None:
                    continue
                if distance >= threshold:
                    actual = 1.0
                else:
                    actual = 0.0
                
                expected = float(test['expected'])
                if expected >= threshold:
                    expected_norm = 1.0
                else:
                    expected_norm = 0.0
                
                if actual == expected_norm:
                    if actual == 1.0:
                        self.detected.append(word)
                        true_positives += 1
                    else:
                        true_negatives += 1
                else:
                    if actual == 1.0:
                        false_positives += 1
                    else:
                        false_negatives += 1
                self.expecteds.append(expected)
                self.actuals.append(actual)
                self.words.append(word)

            dict_distance_metric = {}
            dict_distance_metric['score'] = (true_negatives + true_positives) / (false_negatives + false_positives + true_negatives + true_positives)
            dict_distance_metric['metric'] = distance_metric
            dict_distance_metric['true_positives'] = true_positives
            dict_distance_metric['false_positives'] = false_positives
            dict_distance_metric['true_negatives'] = true_negatives
            dict_distance_metric['false_negatives'] = false_negatives
            
            word2result = {}
            word2expected = {}
            for word, result, expected in list(zip(self.words,self.distances,self.expecteds)):
                word2result[word] = result
                word2expected[word] = expected
            sorted_dict = {k: v for k, v in sorted(word2result.items(), key=lambda item: item[1])}
            top_words = list(sorted_dict.keys())[-10:]
            
            topw2ex = []
            for word in top_words:
                topw2ex.append({word:[word2expected[word],word2result[word]]})
            dict_distance_metric["top_words"] = topw2ex
            dict_distance_metric["detected"] = self.detected
            dict['results'].append(dict_distance_metric)

        with open(f'{dir_name}/words.csv', 'w') as file:
            writer = csv.writer(file)
            for test in test_json['tests']:
                word = test['word']
                embeddings = self.get_embeddings(word)
                for index, embedding in enumerate(embeddings):
                    row = [f'{word}_{index}']
                    for val in embedding:
                        row.append(val)
                    writer.writerow(row)
                
        extra_info = self.save_extra_info(dir_name, test_json["corpora"][0], test_json["corpora"][1])
        if extra_info != None:
            dict.update(extra_info)
        
        # word2rezs = {}
        # for metric in dict['results']:
        #     for word in metric['top_words']:
        #         w = list(word.keys())[0]
        #         word2rezs[w] = word2rezs.get(w, []) + [word[w][1]]
        # metrics = [ result['metric'] for result in dict['results']]
        # with open(f'{dir_name}/{self.get_name()}_{test_json["name"]}.csv', 'w', newline='') as csvfile:
        #     spamwriter = csv.writer(csvfile, delimiter=' ')
        #     spamwriter.writerow(["word"] + metrics)
        #     for word in word2rezs:
        #         plt.bar(metrics, word2rezs[word], align='center')
        #         spamwriter.writerow([word] + word2rezs[word])
        #         plt.savefig(f'{dir_name}/{self.get_name()}_{test_json["name"]}_{word}.png')
        #         plt.close()
            
        desc_file = open(f"{dir_name}/desc.json", "w")
        json.dump(dict, desc_file, indent = 4, ensure_ascii= False)
        desc_file.close()