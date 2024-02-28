import numpy as np
from torch.utils.data import Dataset


class HarDataset(Dataset):
    def __init__(self, dir, trainOrTest):
        self.trainOrTest = trainOrTest
        self.dir = dir
        self.trainData = None
        self.trainLabel = None
        self.testData = None
        self.testLabel = None
        self.initData()


    def __getitem__(self, index):
        if self.trainOrTest == 'train':
            x = self.trainData[index]
            return x, self.trainLabel[index]
        if self.trainOrTest == 'test':
            x = self.testData[index]
            return x, self.testLabel[index]


    def __len__(self):
        if self.trainOrTest == 'train':
            return len(self.trainData)
        if self.trainOrTest == 'test':
            return len(self.testData)

    def initData(self):
        x_train = self.dir + 'x_train.npy'
        x_test = self.dir + 'x_test.npy'
        y_train = self.dir + 'y_train.npy'
        y_test = self.dir + 'y_test.npy'

        self.trainData = np.load(x_train)
        self.trainLabel = np.load(y_train)
        self.testData = np.load(x_test)
        self.testLabel = np.load(y_test)


