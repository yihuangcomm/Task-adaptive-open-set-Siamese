import torch
import torch.nn as nn
import torch.nn.functional as F

class Siamese(nn.Module):

    def __init__(self,feature_n):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            # chanel=1,500,feature
            nn.Conv2d(1, 64, (3,feature_n)),  # 64@498,1 
            nn.MaxPool2d((2,1)),  # 64@249*1  
            nn.Conv2d(64, 128, (16,1)),  # 128@234*1
            nn.ReLU(),    
            nn.MaxPool2d((2,1)),   # 128@117*1 
            nn.Conv2d(128, 128, (16,1)), #128@102*1
            nn.ReLU(), 
            nn.MaxPool2d((2,1)), # 128@51*1
            nn.Conv2d(128, 256, (16,1)), # 256@36*1
            nn.ReLU(),   
        )
        self.liner = nn.Sequential(nn.Linear(9216,4096), nn.Sigmoid())
        self.out = nn.Linear(4096,1)
    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out

if __name__ == '__main__':
    net = Siamese(2)
    print(net)
    print(list(net.parameters()))
