import random
from collections import defaultdict
import pickle as pkl
import torch
import codecs
import csv
import gensim
import re
remove_nonAlphanumeric = re.compile('([^\s\w]|_)+')

# Creates question and token mappings for use in LSTM
# %load C:\Users\sanuj\PycharmProjects\NLP\final\MaLSTM\utils.py
# Creates question and token mappings for use in LSTM
class Dataset:
    def __init__(self, train_file, model=None):
        print('Building data for use in LSTM using', train_file.split('/')[-1])
        word2vec_path = 'GoogleNews-vectors-negative300.bin'
        if not model:
            print('Loading word2vec model')
            self.model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        else:
            self.model = model

        self.stop_words = ['the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'which', 'this', 'that',
                           'these', 'those', 'then',
                           'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while',
                           'during', 'to', 'What', 'Which',
                           'Is', 'If', 'While', 'This']
        # List of (qid1, qid2) instances
        self.train = []
        # List of labels corresponding to train data
        self.labels = []

        # Dict of words to index in the complete data set
        self.vocab = {}
        self.id2word = {}
        # Dict of question id to word index sequence in the question
        self.qid_map = {}

        # Initializes with qid -> list of tokens
        print('Preprocessing sentences..')
        self.initialize_q_map(train_file)
        # Builds vocabulary from tokens in qids_map
        print('Building Vocab...')
        self.build_vocab()
        # Maps questions as sequence of word indices using vocab
        self.map_qids()
        print('Data ready to use.')

    def get_batches(self, batch_size, train_size=None):
        if not train_size:
            train_size = len(self.train)
        combined = list(zip(self.train[:train_size], self.labels[:train_size]))
        random.shuffle(combined)
        self.train[:train_size], self.labels[:train_size] = zip(*combined)
        for start in range(0, train_size, batch_size):
            batch = [(torch.LongTensor(self.qid_map[q1]), torch.LongTensor(self.qid_map[q2])) for q1, q2 in
                     self.train[start:start + batch_size]]
            targets = torch.FloatTensor(self.labels[start:start + batch_size])
            yield batch, targets

    def get_test_data(self, start, total):
        test = [(torch.LongTensor(self.qid_map[q1]), torch.LongTensor(self.qid_map[q2])) for q1, q2 in self.train[start:start+total]]
        targets = torch.FloatTensor(self.labels[start:start + total])
        return test, targets

    def initialize_q_map(self, train_f):
        with codecs.open(train_f, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for row in reader:
                row_id, qid1, qid2, q1, q2, label = row
                qid1, qid2, label = int(qid1), int(qid2), int(label)
                self.labels.append(label)
                self.train.append((qid1, qid2))

                if qid1 not in self.qid_map:
                    self.qid_map[qid1] = self.preprocess_sentence(q1)
                if qid2 not in self.qid_map:
                    self.qid_map[qid2] = self.preprocess_sentence(q2)
        print('Q map initialized!')

    def build_vocab(self):
        not_found = defaultdict(int)
        upper_found = defaultdict(int)
        lower_found = defaultdict(int)
        title_found = defaultdict(int)
        for qid, q in self.qid_map.items():
            for token in q:
                if token not in self.vocab:
                    if token not in self.model.vocab:
                        if token.lower() in self.model.vocab:
                            token = token.lower()
                            lower_found[token] += 1
                        elif token.upper() in self.model.vocab:
                            token = token.upper()
                            upper_found[token] += 1
                        elif token.title() in self.model.vocab:
                            token = token.title()
                            title_found[token] += 1
                        else:
                            not_found[token] += 1
                            continue
                    self.vocab[token] = self.model.vocab[token].index
        print('Vocabulary built! Size:', len(self.vocab))
        print('not found {} upper found {} lower found {} title found {}'.format(len(not_found), len(upper_found), len(lower_found), len(title_found)))

    def map_qids(self):
        for qid, question in self.qid_map.items():
            self.qid_map[qid] = [self.vocab[token] for token in question if token in self.vocab]

    def preprocess_sentence(self, q):
        q = remove_nonAlphanumeric.sub(' ', q)
        q = q.split()
        q = [w for w in q if w not in self.stop_words]
        return q
