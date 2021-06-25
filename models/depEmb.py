import gensim
import pickle as pkl
import codecs
import numpy as np
from gensim.models import Word2Vec
from pathlib import Path

class DepEmbedding(object):
    def __init__(self, args, train_data, test_data, dev_data=None):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.args = args
        self.dep2idx = {}
        self.embedding_dep = None
        
    def get_deps_vocab(self):
        data = []
        vocab = []
        for sentence in self.train_data.deps:
            data.append(sentence)
            vocab.extend(sentence)
        if self.dev_data is not None:
            for sentence in self.dev_data.deps:
                data.append(sentence)
                vocab.extend(sentence)
        for sentence in self.test_data.deps:
            data.append(sentence)
            vocab.extend(sentence)
        return data, list(set(vocab))
    
    def generate_embedding(self, max_features, model_word, vocab):
        embedding = np.zeros([max_features, self.args.dim_emb])
        for index, w in zip(vocab.values(), vocab.keys()):          
            if w in list(model_word.wv.vocab):
                vec = model_word[w]
            else:
                vec = np.random.uniform(-0.25,0.25, self.args.dim_emb)
            embedding[index] = vec
        return embedding
    
    def train(self):
        dep_tac, vocab_dep = self.get_deps_vocab()
        dep2idx = {w: i + 3 for i, w in enumerate(vocab_dep)}
        dep2idx["[CLS]"] = 0
        dep2idx["[SEP]"] = 1
        dep2idx["self"] = 2
        self.dep2idx = dep2idx
        dep2vec = Word2Vec(sentences=dep_tac, size=self.args.dim_emb, window=5, min_count=0, max_vocab_size=len(dep_tac))
        dep2vec.train(dep_tac, total_words=len(dep_tac), epochs=3)
        n_dep = len(dep2idx)+1
        embedding_dep = self.generate_embedding(n_dep, model_word=dep2vec, vocab=dep2idx)
        self.embedding_dep = embedding_dep

        with codecs.open(self.args.dep2idx_path, 'wb') as w:
            pkl.dump(dep2idx, w)

        with codecs.open(self.args.embedding_dep_path, 'wb') as w:
            pkl.dump(embedding_dep, w)

    def load(self):
        with codecs.open(self.args.embedding_dep_path, 'rb') as f: 
            embedding_dep = pkl.load(f)

        with codecs.open(self.args.dep2idx_path, 'rb') as f: 
            dep2idx = pkl.load(f)

        self.embedding_dep = embedding_dep
        self.dep2idx = dep2idx