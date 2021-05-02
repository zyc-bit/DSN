import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time

class BearingDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y


def different_loss(input1, input2):

    inner_loss = torch.inner(input1, input2)
    # batch_size = input1.size(0)
    # input1 = input1.view(batch_size, -1)
    # input2 = input2.view(batch_size, -1)
    #
    # input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
    # input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
    #
    # input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
    # input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)
    #
    # diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

    # return diff_loss, inner_loss
    return inner_loss


def mse(pred, real):
    diffs = torch.add(real, -pred)
    n = torch.numel(diffs.data)
    mse_loss = torch.sum(diffs.pow(2)) / n

    return mse_loss


def simse(pred, real):
    diffs = torch.add(real, - pred)
    n = torch.numel(diffs.data)
    simse_loss = torch.sum(diffs).pow(2) / (n ** 2)

    return simse_loss


# encoder
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv1d(1, 16, 16, stride=8)
        self.BatchNorm1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.maxpooling1 = nn.MaxPool1d(2, 2)
        self.cnn2 = nn.Conv1d(16, 32, 3, 1)
        self.BatchNorm2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.maxpooling2 = nn.MaxPool1d(2, 2)
        self.flat = nn.Flatten()
        self.dense1 = nn.Linear(2496, 512)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.BatchNorm1(x)
        x = self.relu1(x)
        x = self.maxpooling1(x)
        x = self.cnn2(x)
        x = self.BatchNorm2(x)
        x = self.relu2(x)
        x = self.maxpooling2(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.relu3(x)
        return x


# decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cnn1 = nn.Conv1d(1, 16, 16, stride=8)
        self.BatchNorm1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.maxpooling1 = nn.MaxPool1d(2, 2)
        self.cnn2 = nn.Conv1d(16, 32, 3, 1)
        self.BatchNorm2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.maxpooling2 = nn.MaxPool1d(2, 2)
        self.flat = nn.Flatten()
        self.dense1 = nn.Linear(2496, 512)
        self.relu3 = nn.ReLU()

    def forward(self, private, shared):
        pass


class Classifier(nn.Module):
    def __init__(self):
        self.dense1 = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x


train_file = 'Series/Small_500_100/D/TRAIN.npy'
test_file = 'Series/Small_500_100/C/TRAIN.npy'
train_data = np.load(train_file, allow_pickle=True)
test_data = np.load(test_file, allow_pickle=True)
train_simples = train_data.shape[0]
test_simples = test_data.shape[0]
X_train_total = train_data[:, 1:]
Labels_train_total = train_data[:, 0]
X_test = test_data[:, 1:]
Labels_test = test_data[:, 0]
train_y = []
for i in Labels_train_total:
    train_y += [i[2]]
Label_for_train = train_y
test_y = []
for i in Labels_test:
    test_y += [i[2]]
Label_for_test = test_y

X_train_total = X_train_total.astype('float32')
X_train_total = torch.from_numpy(X_train_total)
X_test = X_test.astype('float32')
X_test = torch.from_numpy(X_test)

train_set = BearingDataset(X_train_total, Label_for_train)
test_set = BearingDataset(X_test, Label_for_test)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set)

encoder_source = CNN().cuda()
encoder_target = CNN().cuda()
encoder_shared = CNN().cuda()
decoder = Decoder().cuda()
classifier = Classifier().cuda()

optimizer_for_encoder_source = torch.optim.Adam(encoder_source.parameters())
optimizer_for_encoder_target = torch.optim.Adam(encoder_target.parameters())
optimizer_for_encoder_shared = torch.optim.Adam(encoder_shared.parameters())
optimizer_for_decoder = torch.optim.Adam(decoder.parameters())
optimizer_for_classifier = torch.optim.Adam(classifier.parameters())

num_epoch = 50

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    for i, ((source_data, source_label), (target_data, target_label)) in enumerate(zip(train_loader, test_loader)):
        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()
        target_label = target_label.cuda()
        source_pred_private = encoder_source(source_data)
        source_pred_shared = encoder_shared(source_data)
        different = different_loss(source_pred_private, source_pred_shared)
        different.backward()
        optimizer_for_encoder_source.step()
        optimizer_for_encoder_shared.step()

        recon_for_source = Decoder(private=source_pred_private, shared=source_pred_shared)
        source_recon_loss = simse(recon_for_source, source_data)
        source_recon_loss.backward()
        optimizer_for_decoder.step()

        target_pred_private = encoder_target(target_data)
        target_pred_shared = encoder_shared(target_data)
        different = different_loss(target_pred_private, target_pred_shared)
        different.backward()
        optimizer_for_encoder_target.step()
        optimizer_for_encoder_shared.step()

        recon_for_target = Decoder(private=target_pred_private, shared=target_pred_shared)
        target_recon_loss = simse(recon_for_target, target_data)
        target_recon_loss.backward()
        optimizer_for_decoder.step()

        train_pred = Classifier(target_pred_shared)
        loss = torch.nn.CrossEntropyLoss(train_pred, target_label)
        loss.backward()
        optimizer_for_classifier.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == target_label.numpy())
