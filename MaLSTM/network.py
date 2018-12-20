import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence
import gensim

# LSTM architecture used as a component of the Siamese network
class LSTM(nn.Module):
    def __init__(self, config, model):
        super(LSTM, self).__init__()

        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size

        # make embedding matrix

        # self.word_embs = nn.Embedding(
        #     num_embeddings=self.vocab_size,
        #     embedding_dim=self.embedding_dim, )

        # Load in Word2Vec as embeddings for model
        weights = torch.FloatTensor(model.vectors)[:, :config.embedding_dim]
        self.word_embs = nn.Embedding.from_pretrained(weights)

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim)

    def init_hidden(self):
        # Initialize hidden state (num_layers * num_directions, batch, hidden_size)
        return (torch.randn(1, 1, self.hidden_dim),
                torch.randn(1, 1, self.hidden_dim))

    def forward(self, input, h):
        word_embedding = self.word_embs(input).view(1, 1, -1)
        output, h = self.lstm(word_embedding, h)
        return output, h


class SiameseLSTM(nn.Module):
    def __init__(self, config, model):
        super(SiameseLSTM, self).__init__()

        # TODO configuration
        self.encoder = LSTM(config, model)

    def encoder_params(self):
        return self.encoder.parameters()

    def forward(self, batch):
        h1 = self.encoder.init_hidden()
        h2 = self.encoder.init_hidden()

        pred = torch.zeros(len(batch))
        for i, q_pair in enumerate(batch):
            q1, q2 = q_pair
            for w in q1:
                out1, h1 = self.encoder(w, h1)
            for w in q2:
                out2, h2 = self.encoder(w, h2)
            prediction = torch.exp(-torch.norm((h1[0] - h2[0]), 1))
            pred[i] = prediction

        return pred