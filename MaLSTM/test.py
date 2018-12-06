import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.preprocessing import binarize
import sklearn.metrics as met
# import gensim
# import timeit
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import gc
# from network import SiameseLSTM
# from utils import Dataset
# from config import Config
#
# word2vec_path = "C:\\Users\\sanujb\\PycharmProjects\\CS585_FinalProject\\MaLSTM\\data\\GoogleNews-vectors-negative300.bin"
# model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
# train_data = Dataset('C:\\Users\\sanujb\\PycharmProjects\\CS585_FinalProject\\MaLSTM\\data\\train.csv', model)
# config = Config(len(train_data.vocab))
# rnn = SiameseLSTM(config, train_data.model)
#
# loss_function = nn.BCELoss()
# optimizer = optim.Adadelta(rnn.encoder_params())
# num_epochs = 3
# train_size = 1000
#
# checkpoint = torch.load('C:\\Users\\sanujb\\PycharmProjects\\CS585_FinalProject\\MaLSTM\\data\\model.pt')
# rnn.load_state_dict(checkpoint['rnn_state_dict'])
# rnn.encoder.load_state_dict(checkpoint['encoder_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# config = checkpoint['config']
# test_start = checkpoint['train_size']
#
# test_size = 10000
# test, targets = train_data.get_test_data(test_start, test_size)
# preds = rnn(test, targets)
# print(loss_function(preds, targets))
# correct = 0
# for i in range(test_size):
#     if targets[i] == 1:
#         if preds[i] > 0.5:
#             correct += 1
#     if targets[i] == 0:
#         if preds[i] <= 0.5:
#             correct += 1
# print(correct)
# print(correct/10)
# with open('pred.pkl', 'wb') as file:
#     pkl.dump(preds, file)
# with open('target.pkl', 'wb') as file:
#     pkl.dump(targets, file)

with open('pred.pkl', 'rb') as file:
    preds = pkl.load(file)
with open('target.pkl', 'rb') as file:
    targets = pkl.load(file)

preds = preds.detach().numpy().reshape(-1, 1)
print(preds)
targets = targets.numpy().reshape(-1, 1)
xp = list(_/1000.0 for _ in range(1000))
acc = []
f1 = []
prec = []
recall = []
for thresh in xp:
    x = binarize(preds, thresh, copy=True)
    f1.append(met.f1_score(targets, x))
    acc.append(met.accuracy_score(targets, x))
    prec.append(met.precision_score(targets, x))
    recall.append(met.recall_score(targets, x))

with open('precision.pkl', 'wb') as file:
    pkl.dump(prec, file)
with open('recall.pkl', 'wb') as file:
    pkl.dump(recall, file)
print(max(acc), acc.index(max(acc)), acc[f1.index(max(f1))])
print(max(f1), f1.index(max(f1)), f1[acc.index(max(acc))])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# ax.plot(xp, acc, 'r', xp, f1, 'b')
ax.plot(recall, prec)
plt.show()
