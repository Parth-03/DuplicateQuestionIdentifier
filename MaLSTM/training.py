import gensim
import timeit
import torch
import torch.nn as nn
import torch.optim as optim
from network import SiameseLSTM
from utils import Dataset
from config import Config

word2vec_path = "C:\\Users\sanuj\PycharmProjects\\NLP\\final\MaLSTM\\GoogleNews-vectors-negative300.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
train_data = Dataset('C:\\Users\\sanuj\\PycharmProjects\\NLP\\final\\MaLSTM\data\\train.csv', model)
config = Config(len(train_data.vocab))
rnn = SiameseLSTM(config, train_data.model)

loss_function = nn.BCELoss()
optimizer = optim.Adadelta(rnn.encoder_params())
num_epochs = 1
train_size = 100
batch_size = 64

start_time = timeit.default_timer()
losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    batch_num = 1
    for batch, targets in train_data.get_batches(batch_size, train_size):
        preds = rnn(batch, targets)
        batch_loss = loss_function(preds, targets)
        epoch_loss += batch_loss

        optimizer.zero_grad()  # reset the gradients from the last batch
        batch_loss.backward()  # does backprop!!!
        optimizer.step()  # updates parameters using gradients

        # print('Batch number: {}, batch loss: {}, epoch loss {}'.format(batch_num, batch_loss, epoch_loss))
        batch_num += 1

    losses.append(epoch_loss)
    print(epoch, epoch_loss)

torch.save({
    'epoch': num_epochs,
    'rnn_state_dict': rnn.state_dict(),
    'encoder_state_dict': rnn.encoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': losses,
    'config': config,
    'train_size': train_size,
    'batch_size': batch_size
}, 'C:\\Users\\sanuj\\PycharmProjects\\NLP\\final\\MaLSTM\\data\\model.pt')

print('Total time taken:', timeit.default_timer() - start_time)