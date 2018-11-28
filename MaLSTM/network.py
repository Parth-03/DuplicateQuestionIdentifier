import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence

# LSTM architecture used as a component of the Siamese network
class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()

        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size

        # make embedding matrix
        self.word_embs = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim, )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim)

    def init_hidden(self):
        # Initialize hidden state (num_layers * num_directions, batch, hidden_size)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, input, h):
        word_embedding = self.word_embs(input).view(1, 1, -1)
        output, h = self.lstm(word_embedding, h)
        return output, h


class SiameseLSTM(nn.Module):
    def __init__(self, config):
        super(SiameseLSTM, self).__init__()

        # TODO configuration
        self.encoder = LSTM(config)


    def encoder_params(self):
        return self.encoder.parameters()


    def forward(self, batch, targets):
        batch_size = len(batch)
        h1 = self.encoder.init_hidden()
        h2 = self.encoder.init_hidden()

        pred = []
        for q_pair in batch:
            q1, q2 = q_pair
            for w in q1:
                out1, h1 = self.encoder(w, h1)
            for w in q2:
                out2, h2 = self.encoder(w, h2)
            prediction = torch.exp(-torch.norm((h1 - h2), 1))
            pred.append(prediction)

