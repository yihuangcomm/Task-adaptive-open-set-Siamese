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

    def __init__(self, dataPath, transform=None, features='0,20', dropout_p=0.0, dropout_n=0.6):
        super(EpnetworkTrain, self).__init__()
        np.random.seed(0)
        self.transform = transform
        if features!='full':
            self.features = features.split(',')
        else:
            self.features = features
        self.datas, self.num_classes, self.label_list = self.loadToMem(dataPath)
        self.dropout_p = dropout_p
        self.dropout_n = dropout_n

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        X = {}
        idx = 0
        hidx_list = []
        for classPath in os.listdir(dataPath):
            X[idx] = []
            for samplePath in os.listdir(os.path.join(dataPath, classPath)):
                x = np.array(pandas.read_csv(os.path.join(dataPath, classPath,samplePath)).loc[:])
                if self.features!='full':
                    self.features = [int(i) for i in self.features]
                    x = x[:,self.features] #[0,20] or[0]
                X[idx].append(x)
            # set coarse labels
            if samplePath[0]=='Y': # video streams from youtube platform
                hidx_list.append(0.0)
            elif samplePath[0]=='N': # video streams from netflix platform
                hidx_list.append(1.0)
            else:                     # video streams from stan platform
                hidx_list.append(2.0)
            idx += 1
        print("finish loading training dataset to memory")
        return X, idx, hidx_list
      

    def __len__(self):
        return 21000000

    def __getitem__(self, index):
        label = None
        label_1 = None
        nt1 = None
        nt2 = None
        # get samples from the same class
        random.seed(index)
        if index % 2 == 1:
            label = 1.0
            label_1 = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            nt1 = random.choice(self.datas[idx1][0:80]) # left 20% training samples for test if necessary. we didn't use it.
            # get positive sample nt2 by dropout the nt1 with dropout_p
            if (self.dropout_p != 0.0) & (index%4!=1):
                row, col = len(nt1),len(nt1[0])
                nt2 = nt1*np.random.binomial([np.ones((row,col))],1-self.dropout_p)[0] * (1.0/(1-self.dropout_p))
            else:
                nt2 = random.choice(self.datas[idx1][0:80])
                
        # get samples from different classes
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            h_idx1 = self.label_list[idx1]
            nt1 = random.choice(self.datas[idx1][0:80])
            # get negative sample nt2 by dropout the nt1 with dropout_n
            if (index % 4 != 0) & (self.dropout_n!=0.0):
                row, col = len(nt1),len(nt1[0])
                nt2 = nt1*np.random.binomial([np.ones((row,col))],1-self.dropout_n)[0] * (1.0/(1-self.dropout_n))
                label_1 = 1.0
            else:
                idx2 = random.randint(0, self.num_classes - 1)
                while idx1 == idx2:
                    idx2 = random.randint(0, self.num_classes - 1)               
                nt2 = random.choice(self.datas[idx2][0:80])
                h_idx2 = self.label_list[idx2]
                if h_idx1==h_idx2:
                    label_1 = 1.0
                else:
                    label_1 = 0.0

        nt1 = torch.tensor(nt1, dtype=torch.float32)
        nt2 = torch.tensor(nt2, dtype=torch.float32)
        
        return nt1, nt2, torch.from_numpy(np.array([label], dtype=np.float32)), torch.from_numpy(np.array([label_1], dtype=np.float32))


class EpnetworkTest(Dataset):

    def __init__(self, dataPath, transform=None, times=500, way=10,  features='0,20'):
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
        label = None
        idx = index % (self.way)
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
