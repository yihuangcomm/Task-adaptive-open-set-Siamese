import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image


class OmniglotTrain(Dataset):

    def __init__(self, dataPath, transform=None, features='2', dropout_p=0.0, dropout_n=0.0):
        super(OmniglotTrain, self).__init__()
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
        datas = {}
        degrees = [0,90,180,270] # distortion
        idx = 0 # fine label
        hidx = 0
        hidx_list = [] # coarse label
        for degree in degrees:
            hidx = 0
            for alphaPath in os.listdir(dataPath):                
                for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                    datas[idx] = []
                    for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                        filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                        datas[idx].append(Image.open(filePath).rotate(degree).convert('L'))
                    idx += 1
                    hidx_list.append(hidx)
                hidx += 1
        print("finish loading training dataset to memory")
        return datas, idx, hidx_list

    def __len__(self):
        return  21000000

    def __getitem__(self, index):
        label = None
        label_1 = None
        image1 = None
        image2 = None
        # get image from same class
        random.seed(index) # make sure that results are reproductable.
        if index % 2 == 1:
            label = 1.0
            label_1 = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            # get positive sample image2 by dropout the image1 with dropout_p
            if (self.dropout_p != 0.0) & (index%4!=1):
                row, col = 105, 105 #len(image1),len(image1[0])
                image2 = np.asarray(image1)*np.random.binomial([np.ones((row,col))],1-self.dropout_p)[0] * (1.0/(1-self.dropout_p))
                image2 = Image.fromarray(image2)
            else:
                image2 = random.choice(self.datas[idx1])
                
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            h_idx1 = self.label_list[idx1]
            image1 = random.choice(self.datas[idx1])
            # get negative sample image2 by dropout the image1 with dropout_n
            if (index % 4 != 0) & (self.dropout_n!=0.0):
                row, col = 105, 105 #len(image1),len(image1[0])
                image2 = np.asarray(image1)*np.random.binomial([np.ones((row,col))],1-self.dropout_n)[0] * (1.0/(1-self.dropout_n))
                image2 = Image.fromarray(image2)
                label_1 = 1.0
            else:
                idx2 = random.randint(0, self.num_classes - 1)
                while idx1 == idx2:
                    idx2 = random.randint(0, self.num_classes - 1)               
                image2 = random.choice(self.datas[idx2])
                h_idx2 = self.label_list[idx2]
                if h_idx1==h_idx2:
                    label_1 = 1.0
                else:
                    label_1 = 0.0

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32)), torch.from_numpy(np.array([label_1], dtype=np.float32))



class OmniglotTest(Dataset):
    def __init__(self, dataPath, transform=None, times=500, way=10, features='2'):
        np.random.seed(1)
        super(OmniglotTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        if features!='full':
            self.features = features.split(',')
        else:
            self.features = features
        self.datas, self.num_classes, self.label_list = self.loadToMem(dataPath)
        self.target_list = random.sample(range(self.num_classes),self.way)
        
    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        datas = {}
        idx = 0 # fine label
        hidx = 0
        hidx_list = [] # coarse label
        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    datas[idx].append(Image.open(filePath).convert('L'))
                idx += 1
                hidx_list.append(hidx)
            hidx += 1
        print("finish loading test dataset to memory")
        return datas, idx, hidx_list

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way # generate test/validation batches for each query task.
        label = None
        random.seed(index)
        if idx == 0:
            self.target_list = random.sample(range(self.num_classes),self.way) # generate support classes id randomly
            if index % (2*self.way) ==0:
                # generate closed-set test batch pairs
                label = 1.0
                self.c1 = self.target_list[0]
                self.img1 = random.choice(self.datas[self.c1])
                random.seed(index+1)
                img2 = random.choice(self.datas[self.c1])
            else:
                # generate open-set test batch pairs
                label = 0.0
                self.c1 = self.target_list[0]
                self.img1 = random.choice(self.datas[self.c1])
                #repeatly use one of support class to construct open-set query task, to make sure every pair is negative
                c2 = self.target_list[self.way-1] 
                img2 = random.choice(self.datas[c2])
                
        # generate image pair from different class
        else:
            label = 0.0
            c2 = self.target_list[idx]
            img2 = random.choice(self.datas[c2])

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(np.array([label])).type(torch.FloatTensor)


if __name__=='__main__':
    omniglotTrain = OmniglotTrain('../omniglot/python/images_background')
    print(omniglotTrain)
