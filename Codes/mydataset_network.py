import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
from PIL import Image
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class EpnetworkTrain(Dataset):

    def __init__(self, dataPath, transform=None, features='0,20'):
        super(EpnetworkTrain, self).__init__()
        np.random.seed(0)
        self.transform = transform
        if features!='full':
            self.features = features.split(',')
        else:
            self.features = features
        self.datas, self.num_classes = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        X = {}
        idx = 0
        for classPath in os.listdir(dataPath):
            X[idx] = []
            for samplePath in os.listdir(os.path.join(dataPath, classPath)):
                x = np.array(pandas.read_csv(os.path.join(dataPath, classPath,samplePath)).loc[:])
                if self.features!='full':
                    self.features = [int(i) for i in self.features]
                    x = x[:,self.features] #[0,20] or[0]
                X[idx].append(x)
            idx += 1               
        print("finish loading training dataset to memory")
        return X, idx
      

    def __len__(self):
        return 21000000

    def __getitem__(self, index):
        label = None
        nt1 = None
        nt2 = None
        # get samples from the same class
        random.seed(index)
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            nt1 = random.choice(self.datas[idx1][0:80])  #left 20% of them for test if necessary
            nt2 = random.choice(self.datas[idx1][0:80])
        # get samples from different classes
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            nt1 = random.choice(self.datas[idx1][0:80])
            nt2 = random.choice(self.datas[idx2][0:80])
            
        nt1 = torch.tensor(nt1, dtype=torch.float32)
        nt2 = torch.tensor(nt2, dtype=torch.float32)
        return nt1, nt2, torch.from_numpy(np.array([label])).type(torch.FloatTensor)


class EpnetworkTest(Dataset):

    def __init__(self, dataPath, transform=None, times=600, way=10, features='0,20'):
        np.random.seed(1)
        super(EpnetworkTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.nt1 = None
        self.c1 = None
        if features!='full':
            self.features = features.split(',')
        else:
            self.features = features
        self.datas, self.num_classes = self.loadToMem(dataPath)
        

    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        X = {}
        idx = 0
        for classPath in os.listdir(dataPath):
            X[idx] = []
            for samplePath in os.listdir(os.path.join(dataPath, classPath)):
                x = np.array(pandas.read_csv(os.path.join(dataPath, classPath,samplePath)).loc[:])
                if self.features!='full':
                    self.features = [int(i) for i in self.features]
                    x = x[:,self.features] #[0,20] or[0]
                X[idx].append(x)
            idx += 1       
        print("finish loading test dataset to memory")
        return X, idx

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        random.seed(index)
        if idx == 0:
            self.target_list = random.sample(range(self.num_classes),self.way)
            if index % (2*self.way) ==0:
                # generate closed-set test batch pairs
                label = 1.0
                self.c1 = self.target_list[0]
                self.nt1 = random.choice(self.datas[self.c1])
                random.seed(index+1)
                nt2 = random.choice(self.datas[self.c1])
            else:
                # generate open-set test batch pairs
                label = 0.0
                self.c1 = self.target_list[0]
                self.nt1 = random.choice(self.datas[self.c1])
                #repeatly use one of support class to construct open-set query task, to make sure every pair is negative 
                c2 = self.target_list[self.way-1] 
                nt2 = random.choice(self.datas[c2])

        # generate sample pairs from different classes
        else:
            label = 0.0
            c2 = self.target_list[idx]
            nt2 = random.choice(self.datas[c2])

        nt1 = torch.tensor(self.nt1, dtype=torch.float32)
        nt2 = torch.tensor(nt2, dtype=torch.float32)

        return nt1, nt2, torch.from_numpy(np.array([label])).type(torch.FloatTensor)


if __name__=='__main__':
    EpnetworkTrain = EpnetworkTrain('../train_train')
    print(EpnetworkTrain)
