import argparse
import os.path
from datetime import datetime
import time
import math
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
from dataset import HarDataset
from torch.utils.data import DataLoader
from models.MFCANN import MFCANN
from utils import EarlyStopping, cal_best_model_evaluating_indicator


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--dataset-number', default=1, type=int, help='dataset label', dest='dataset_number')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'use device: {device}')

criterion = nn.CrossEntropyLoss().cuda(device)

dataset_dict = {
    'uci-har': {
        'file_path': './data/UCI_data/np/',
        'd_input': 128,
        'd_channel': 9,
        'd_output': 6,
    },
    'usc-had': {
        'file_path': './data/USC_HAD_data/np/',
        'd_input': 128,
        'd_channel': 6,
        'd_output': 12,
    },
    'real-world': {
        'file_path': './data/Real_World_data/np/',
        'd_input': 128,
        'd_channel': 21,
        'd_output': 8,
    }
}
dataset_names = ['uci-har', 'usc-had', 'real-world']


dataset_number = args.dataset_number
use_dataset_name = dataset_names[dataset_number]


dataset_path = dataset_dict[use_dataset_name]['file_path']


batch_size = 32
epochs = 5000
lr = 0.0001
d_input = dataset_dict[use_dataset_name]['d_input']
d_channel = dataset_dict[use_dataset_name]['d_channel']
d_output = dataset_dict[use_dataset_name]['d_output']


seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


test_dataset = HarDataset(dataset_path, trainOrTest='test')
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)
train_dataset = HarDataset(dataset_path, trainOrTest='train')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

train_date = datetime.now()
train_date = train_date.strftime('%Y-%m-%d %H-%M-%S')
train_name = 'dataset_name=' + use_dataset_name + '-seed=' + str(seed) + '-train_date=' + train_date

correct_on_train = []
correct_on_test = []

loss_list = []

test_interval = 1

def train(model):
    model.cuda(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    max_accuracy = 0

    early_stop = EarlyStopping('test_acc', mode='max', patience=50)

    log_interval = int(len(train_dataloader) / batch_size / 5)
    if log_interval == 0:
        log_interval = 1
    for index in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        mean_loss = torch.zeros(1).to(device)
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = x.permute([0, 2, 1])
            x = x.float()
            output = model(x.to(device))
            y = y.long()
            loss = criterion(output, y.to(device))
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

            mean_loss = (mean_loss * i + loss.detach()) / (i + 1)

            total_loss += loss.item()

            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.6f} | {:5.2f} ms | '
                      'loss {:5.5f} | ppl {:8.2f}'.format(
                    index, i, len(train_dataloader) // batch_size, lr,
                              elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()


        if ((index + 1) % test_interval) == 0:
            current_accuracy = evaluate(model, test_dataloader, 'test_set')
            evaluate(model, train_dataloader, 'train_set')
            print(f'Current Maximum Accuracy\tTest Set:{max(correct_on_test)}%\tTrain Set:{max(correct_on_train)}%')
            if early_stop(current_accuracy):
                print("Early Stop!!!!!")
                print(f'Accuracy:%.4f' % max(correct_on_test))
                cal_best_model_evaluating_indicator(model_path='saved_model/' + train_name + '.pth', model=model, testDataloader=test_dataloader, device=device)
                return
            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                if os.path.exists('saved_model') is False:
                    os.mkdir('saved_model')
                torch.save(model.state_dict(), 'saved_model/' + train_name + '.pth')

def evaluate(decoder_model, dataloader, flag='test_set'):
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        decoder_model.eval()
        if flag == 'train_set':
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                x = x.permute([0, 2, 1])
                x = x.float()
                y = y.long()
                y_pre = decoder_model(x)

                loss = criterion(y_pre, y)
                total_loss += loss

                _, label_index = torch.max(y_pre.data, dim=-1)
                total += label_index.shape[0]
                correct += (label_index == y.long()).sum().item()
        elif flag == 'test_set':
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                x = x.permute([0, 2, 1])
                x = x.float()
                y = y.long()
                y_pre = decoder_model(x)

                loss = criterion(y_pre, y)
                total_loss += loss

                _, label_index = torch.max(y_pre.data, dim=-1)
                total += label_index.shape[0]
                correct += (label_index == y.long()).sum().item()
        if flag == 'test_set':
            correct_on_test.append(correct / total)
            mean_loss = total_loss / len(dataloader)
            print(f'Loss on {flag}: %.5f ' % mean_loss)
        elif flag == 'train_set':
            correct_on_train.append(correct / total)

        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))
        return correct / total


if __name__ == "__main__":
    model = MFCANN(in_channels=d_channel, out_channels=d_output, n_filters=32, kernel_sizes=[9, 19, 39],
            bottleneck_channels=32, intra_reduction_radio=1, inter_reduction_radio=4, activation=nn.ReLU(), device=device)

    train(model)