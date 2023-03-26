from representations.representation import Representation
from representations.distances import compute_distance_for_points
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

class SGNS(Representation):
    def get_name(self):
         return "sgns_wi"

    def distance_metrics(self):
        return ['euclidean', 'cosine', 'canberra', 'manhattan', 'braycurtis', 'correlation']

    def load_corpus(self, path, targets, corpus_id):
        data_file = open(path)
        text = data_file.read()
        
        clean_text = text.replace("\n", " ")
        
        sentences = []

        for i in sent_tokenize(clean_text):
            temp = []     
            for j in word_tokenize(i):
                if j.lower() in targets:
                    temp.append(f'{j}{corpus_id}'.lower())
                else:
                    temp.append(j.lower())
            sentences.append(temp)

        return sentences

    def load_data(self, path1, path2, targets):
        self.sentences = self.load_corpus(path1, targets, 1)
        self.sentences += self.load_corpus(path2, targets, 2)
    
    def train(self):
        self.model = Word2Vec(self.sentences, min_count = 1,
                              vector_size = 100, window = 5, sg = 1)
        self.model.train(self.sentences, total_examples = len(self.sentences),
                         epochs = 5)

    def load_model(self, path):
        self.model = Word2Vec.load(path)
    
    def save_model(self, path):
        self.model.save(path)

    def compare(self, word, distance_type):
        ws = self.get_embeddings(word)
        if len(ws) == 2:
            w1 = ws[0]
            w2 = ws[1]
            return compute_distance_for_points(w1, w2, distance_type)
        return None
    
    def get_embeddings(self, word):
        word1 = f"{word}1"
        word2 = f"{word}2"
        if word1 in self.model.wv.key_to_index and word2 in self.model.wv.key_to_index:
            w1 = self.model.wv[word1]
            w2 = self.model.wv[word2]
            return [w1,w2]
        return []