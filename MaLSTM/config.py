class Config:
    def __init__(self, vocab_size):
        self.train_file = ''
        self.test_file = ''

        self.max_sentence_length = 30

        self.embedding_dim = 300
        self.hidden_dim = 50
        self.batch_size = 16
        self.vocab_size = vocab_size