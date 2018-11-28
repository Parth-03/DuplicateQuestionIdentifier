import codecs
import csv

# Creates question and token mappings for use in LSTM
class Dataset:
    def __init__(self, train_file):
        # List of (qid1, qid2) instances
        self.train = []
        # List of labels corresponding to train data
        self.labels = []

        # Dict of words to index in the complete data set
        self.vocab = {'<pad>': 0}
        # Dict of question id to word index sequence in the question
        self.qid_map = {}

        print('Building data for use in LSTM using', train_file.split('/')[-1])
        # Initializes with qid -> list of tokens
        self.initialize_q_map(train_file)
        # Builds vocabulary from tokens in qids_map
        self.build_vocab()
        #         # Maps questions as sequence of word indices using vocab
        #         self.map_qids()
        print('Data ready to use.')


    def get_batches(self, batch_size):
        for start  in range(0, len(self.train), batch_size):
            yield start, start + batch_size


    def initialize_q_map(self, train_f):
        with codecs.open(train_f, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            cols = next(reader)
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
        for qid, q in self.qid_map.items():
            for pos, token in enumerate(q):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        print('Vocabulary built!')

    def map_qids(self):
        for qid, question in self.qid_map.items():
            self.qid_map[qid] = [self.vocab[token] for token in question]

    def preprocess_sentence(self, q):
        return q.split()
