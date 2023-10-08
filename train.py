from __future__ import print_function
import torch
import os
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneOut
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from metrics import Flores, Pillai, Brajdic
from models import attention

from utils import normalize, flipSignals

torch.autograd.set_detect_anomaly(True)

num_epochs = 300  # 250
batch_size = 16
input_dim = 1
hidden_dim = 128
layer_dim = 1
output_dim = 1

lr = 0.005

seed = 69
torch.manual_seed(seed)
np.random.seed(seed)

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
print("Device being used:", device, flush=True)

root = os.curdir
ptFileName = 'WeAllWalk.pt'
filePath = os.path.join(root, ptFileName)

(signals, lengths, steps) = torch.load(filePath)

signals = normalize(signals, lengths)
signals = np.expand_dims(signals, axis=2)

x_train, x_test, y_train, y_test, l_train, l_test = train_test_split(signals, steps, lengths,
                                                                     test_size=.20, random_state=seed)
dataset_sizes = {'train': x_train.shape[0],
                 'test': x_test.shape[0]}

train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train), torch.Tensor(l_train))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test), torch.Tensor(l_test))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
stratifiedKFold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

criterion = nn.MSELoss(reduction='mean').to(device)
softmax = nn.Softmax()
sigmoid = nn.Sigmoid()

Y_TRUES = np.empty([0])
Y_PREDS = np.empty([0])

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(train_ids))
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(test_ids))

    dataloader = {'train': train_dataloader,
                  'test': test_dataloader}

    model = attention(input_dim, hidden_dim, output_dim).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=75, gamma=.1)

    for epoch in range(num_epochs):

        for phase in ['train', 'test']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = .0

            y_trues = np.empty([0])
            y_preds = np.empty([0])

            for inputs, labels, lengths in tqdm(dataloader[phase], disable=True):

                inputs = inputs.to(device)
                labels = labels.to(device)
                lengths = lengths.long().to(device)

                labels = labels.float()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, lengths)
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                y_trues = np.append(y_trues, labels.cpu().detach().numpy())
                y_preds = np.append(y_preds, outputs.cpu().detach().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                scheduler.step()

            print("[{}] Fold: {}/{} Epoch: {}/{} Loss: {} LR: {}".format(
                phase, fold + 1, k_folds, epoch + 1, num_epochs, epoch_loss, scheduler.get_last_lr()), flush=True)

            if phase == 'test' and epoch == num_epochs - 1:
                np.savetxt('y_trues' + str(fold) + '.csv', y_trues, delimiter=',')
                np.savetxt('y_preds' + str(fold) + '.csv', y_preds, delimiter=',')
                Y_TRUES = np.append(Y_TRUES, y_trues)
                Y_PREDS = np.append(Y_PREDS, y_preds)

np.save('y_trues' + '.npy', Y_TRUES)
np.save('y_preds' + '.npy', Y_PREDS)


print(mean_absolute_error(Y_TRUES, Y_PREDS))
print(Flores(Y_TRUES, Y_PREDS))
print(Pillai(Y_TRUES, Y_PREDS, mean=True))
print(Brajdic(Y_TRUES, Y_PREDS, median=True))
