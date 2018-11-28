import torch.nn as nn
import torch.optim as optim
from network import LSTM, SiameseLSTM
from utils import Dataset
from config import Config


train_data = Dataset('data/train.csv')
config = Config(len(train_data.vocab))
rnn = SiameseLSTM(config)

loss_function = nn.MSELoss()
optimizer = optim.Adadelta(rnn.encoder_params())
num_epochs = 10

for epoch in range(num_epochs):
    epoch_loss = 0
    for start, end in train_data.get_batches(config.batch_size):
        pass