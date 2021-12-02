import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):

    def __init__(self,feature_n):
        super(Siamese, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 64@48*48
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    # 128@42*42
            nn.MaxPool2d(2)   # 128@21*21
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 128, 4),
            nn.ReLU(), # 128@18*18
            nn.MaxPool2d(2) # 128@9*9
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),   # 256@6*6
        )

        self.liner_1 = nn.Sequential(nn.Linear(10368, 4096), nn.Sigmoid())
        self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid()) 
        self.out = nn.Linear(4096, 1)
        self.out_1 = nn.Linear(4096,1)

    def forward_one(self, x, is_hloss):     
        x = self.conv_1(x)
        x = self.conv_2(x)
        x_1 = self.conv_3(x)    
        x = self.conv_4(x_1)            
        if is_hloss==True:
            x_1 = x_1.view(x_1.size()[0], -1)
            x_1 = self.liner_1(x_1)  # coarse level feature
        x = x.view(x.size()[0], -1)
        x = self.liner(x)  #fine level feature
        if is_hloss==True:
            return x_1, x
        else:
            return x, x
        
    def forward(self, x1, x2,is_hloss=True):
        out1_1, out1 = self.forward_one(x1,is_hloss)
        out2_1, out2 = self.forward_one(x2,is_hloss)
       
        dis_1 = torch.abs(out1_1 - out2_1) # for hierarchical cross entropy loss
        dis = torch.abs(out1 - out2)
        
        out_1 = self.out_1(dis_1)
        out = self.out(dis)

        return out_1, out

# for test
if __name__ == '__main__':
    net = Siamese()
    print(net)
    print(list(net.parameters()))
