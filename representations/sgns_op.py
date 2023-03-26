from representations.representation import Representation
from representations.distances import compute_distance_for_points
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from copy import deepcopy
import numpy as np
import re

class SGNS(Representation):
    def get_name(self):
         return "sgns_op"

    def distance_metrics(self):
        return ['euclidean', 'cosine', 'canberra', 'manhattan', 'braycurtis', 'correlation']

    def load_corpus(self, path, _):
        data_file = open(path)
        text = data_file.read()
        
        regex = re.compile('([^\s\w]|_)+')
        clean_text = text.replace("\n", " ")
        clean_text = regex.sub('', clean_text)

        sentences = sent_tokenize(clean_text)
        sentences = []

        for i in sent_tokenize(clean_text):
            temp = []     
            for j in word_tokenize(i):
                temp.append(j.lower())
            sentences.append(temp)

        return sentences

    def load_data(self, path1, path2, targets):
        self.sentences1 = self.load_corpus(path1, targets)
        self.sentences2 = self.load_corpus(path2, targets)
    
    def intersection_align_gensim(self):
        vocab1 = set(self.model1.wv.index_to_key)
        vocab2 = set(self.model2.wv.index_to_key)

        self.vocab1 = deepcopy(vocab1)
        self.vocab2 = deepcopy(vocab2)

        common_vocab = vocab1 & vocab2
        
        if not vocab1 - common_vocab and not vocab2 - common_vocab:
            return

        common_vocab = list(common_vocab)
        common_vocab.sort(key=lambda w: self.model1.wv.get_vecattr(w, "count") + self.model2.wv.get_vecattr(w, "count"), reverse=True)
        for m in [self.model1, self.model2]:
            indices = [m.wv.key_to_index[w] for w in common_vocab]
            old_arr = m.wv.vectors
            new_arr = np.array([old_arr[index] for index in indices])
            m.wv.vectors = new_arr
            new_key_to_index = {}
            new_index_to_key = []
            for new_index, key in enumerate(common_vocab):
                new_key_to_index[key] = new_index
                new_index_to_key.append(key)
            m.wv.key_to_index = new_key_to_index
            m.wv.index_to_key = new_index_to_key
            
    def train(self):
        self.model1 = Word2Vec(self.sentences1, min_count = 1,
                              vector_size = 100, window = 5, sg = 1)
        self.model1.train(self.sentences1, total_examples = len(self.sentences1),
                         epochs = 5)
        self.model2 = Word2Vec(self.sentences2, min_count = 1,
                              vector_size = 100, window = 5, sg = 1)
        self.model2.train(self.sentences2, total_examples = len(self.sentences2),
                         epochs = 5)
        self.intersection_align_gensim()

        base_vecs = self.model1.wv.get_normed_vectors()
        other_vecs = self.model2.wv.get_normed_vectors()

        m = other_vecs.T.dot(base_vecs) 
        u, _, v = np.linalg.svd(m)

        ortho = u.dot(v)

        self.model2.wv.vectors = (self.model2.wv.vectors).dot(ortho)

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
        if word in self.vocab1 and word in self.vocab2:
            w1 = np.array(self.model1.wv[word])
            w2 = np.array(self.model2.wv[word])
            return [w1,w2]
        return []