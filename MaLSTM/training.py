import gensim
import timeit
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from network import SiameseLSTM
from utils import Dataset
from config import Config

word2vec_path = "C:\\Users\\sanujb\\PycharmProjects\\CS585_FinalProject\\MaLSTM\\data\\GoogleNews-vectors-negative300.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
train_data = Dataset('C:\\Users\\sanujb\\PycharmProjects\\CS585_FinalProject\\MaLSTM\\data\\train.csv', model)
config = Config(len(train_data.vocab))
rnn = SiameseLSTM(config, train_data.model)

loss_function = nn.BCELoss()
optimizer = optim.Adadelta(rnn.encoder_params())
num_epochs = 100
train_size = 80000

for epoch in range(num_epochs):
    start_time = timeit.default_timer()
    epoch_loss = 0
    batch_num = 1
    for batch, targets in train_data.get_batches(config.batch_size, train_size):
        preds = rnn(batch, targets)
        batch_loss = loss_function(preds, targets)
        epoch_loss += batch_loss

        optimizer.zero_grad()  # reset the gradients from the last batch
        batch_loss.backward()  # does backprop!!!
        torch.nn.utils.clip_grad_norm_(rnn.encoder_params(), 0.25)
        optimizer.step()  # updates parameters using gradients

        if batch_num % 500 == 0:
            print('Batch number: {}, batch loss: {}, epoch loss {}'.format(batch_num, batch_loss, epoch_loss))
        batch_num += 1

    print(epoch, epoch_loss)
    print(timeit.default_timer() - start_time)
    if (epoch+1) % 5 == 0:
        torch.save({
            'epoch': num_epochs,
            'rnn_state_dict': rnn.state_dict(),
            'encoder_state_dict': rnn.encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'train_size': train_size
        }, 'C:\\Users\\sanujb\\PycharmProjects\\CS585_FinalProject\\MaLSTM\\data\\model3.pt')
#
# rnn.encoder.eval()
# test, targets = train_data.get_test_data(5000,1000)
# preds = rnn(test, targets)
# print(loss_function(preds, targets))
# # print(nn.NLLLoss(preds, targets))
# print(preds)
# print(targets)
