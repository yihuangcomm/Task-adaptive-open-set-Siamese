import torch
import pickle
import torchvision
from torchvision import transforms
from mydataset_network import EpnetworkTrain, EpnetworkTest
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model_network import Siamese
import time
import numpy as np
import gflags
import sys
import os
import math
from scipy.special import softmax
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':

    Flags = gflags.FLAGS
    gflags.DEFINE_bool("cuda", True, "use cuda")
    gflags.DEFINE_integer("feature_n", 2, "how many features are chose")
    gflags.DEFINE_string("features", '0,20',  "which features are chose, it can be 'full' which means using all features.")
    gflags.DEFINE_string("train_path", "../train_train", "training folder")
    gflags.DEFINE_string("test_path", "../test", 'path of testing folder')
    gflags.DEFINE_integer("way", 5, "the N ways of 'N way K shot learning'")
    gflags.DEFINE_integer("val_way", 3, "validation based on N ways")
    gflags.DEFINE_float("dropout_p", 0.0, "positive dropout probability")  
    gflags.DEFINE_float("dropout_n", 0.0, "negative dropout probability")
    gflags.DEFINE_float("hloss_alpha", 0.0, "coarse loss factor") 
    gflags.DEFINE_string("times", 600, "number of samples to test accuracy")
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size", 128, "number of batch size") 
    gflags.DEFINE_float("lr", 0.00006, "learning rate") 
    gflags.DEFINE_integer("show_every", 10, "show result after each show_every iter.")
    gflags.DEFINE_integer("save_every", 100, "save model after each save_every iter.")
    gflags.DEFINE_integer("test_every", 100, "test model after each test_every iter.")
    gflags.DEFINE_integer("max_iter", 20000, "number of iterations before stopping")
    gflags.DEFINE_string("model_path", "../siamese_model_40_10_10_new", "path to store model")
    gflags.DEFINE_string("gpu_ids", "0,1,2,3", "gpu ids used to train")
    gflags.DEFINE_string('checkbreakpoint','3200', "checkpoint batch")
    gflags.DEFINE_bool('resume', True, "training from checkpoint") 

    Flags(sys.argv)

    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    feature_n = Flags.feature_n
    features = Flags.features
    way_list = [3, Flags.way, 2*Flags.way] 
    os.environ["CUDA_VISIBLE_DEVICES"] = Flags.gpu_ids
    print("use gpu:", Flags.gpu_ids, "to train.")

    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
    net = Siamese(feature_n)
    optimizer = torch.optim.Adam(net.parameters(),lr = Flags.lr )

    # multi gpu
    if len(Flags.gpu_ids.split(",")) > 1:
        net = torch.nn.DataParallel(net)

    if Flags.cuda:
        net.cuda()
        
    def evaluation(data_loader,feature_n,way_id,filenames):
        net.eval()
        right, error = 0, 0
        soft_score = []
        gt = []
        for _, (test1, test2, label) in enumerate(data_loader, 1):
            c = test1.size(-3)
            h = test1.size(-2)
            w = test1.size(-1)
            test1 = test1.view(c, -1, h, w)
            test2 = test2.view(c, -1, h, w)
            if Flags.cuda:
                test1, test2 = test1.cuda(), test2.cuda()
            test1, test2 = Variable(test1), Variable(test2)
            output = net.forward(test1, test2)
            output = output.data.cpu()
            output = output.numpy()
            gt.append(np.max(label.cpu().numpy()))
            soft_score.append(np.max(output))
            pred = np.argmax(output)
            if np.max(label.cpu().numpy())== 1.0:
                if pred == 0:
                    right += 1
                else: 
                    error += 1
        acc = right*1.0/(right+error)
        try:
            auc = roc_auc_score(np.asarray(gt,dtype=np.float32), np.asarray(soft_score,dtype=np.float32))
        except:
            auc = 0.0
        print('*'*70)
        print('\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(right, error, right*1.0/(right+error)))
        print('*'*70)
        with open(filenames, 'a') as f:
            f.write('current_acc:'+str(acc)+'\n')
            f.write('auc_softscore:'+str(auc)+'\n')
        return acc, auc

    models_list = [int(Flags.checkbreakpoint)]*1
    ACC = {}
    AUC = {}
    for n in range(len(way_list)):
        ACC[n] = []
        AUC[n] = []
    for model_id in models_list:
        #loading checkpoint
        if Flags.resume:
            print('-----------check point loading------------------')
            path_checkpoint = Flags.model_path +'/model-conv4-dropout' + str(Flags.dropout_p) +'_' +str(Flags.dropout_n)+'-'+features+'feature-new-1.0+hloss_' + str(Flags.hloss_alpha) +'-' + str(model_id) + ".pt"
            checkpoint = torch.load(path_checkpoint)         
            net.load_state_dict(checkpoint['model_state_dict']) 
            acc_best_single = checkpoint['acc_best_single']
            test_batch = checkpoint['batch']
            print('------------finished-----------------')
        print("test_epoch:",test_batch)
        i=0
        for ways in way_list: 
            testSet = EpnetworkTest(Flags.test_path, times = Flags.times*2, way = ways, features=features)
            testLoader = DataLoader(testSet, batch_size=ways, shuffle=False, num_workers=Flags.workers)
            acc, auc = evaluation(testLoader,feature_n,i,'conv4_randomway_nodropout_nohloss_'+ str(ways)+'way_'+ features +'_feature' + '_val'+'_based_'+str(Flags.val_way)+'way_single')
            ACC[i].append(acc)
            AUC[i].append(auc)
            i = i+1
    for i in range(len(way_list)):
        print(' %d way:ACC_ave: %f, ACC_std: %f, AUC_ave: %f, AUC_std: %f'%(way_list[i],np.mean(ACC[i]),np.std(ACC[i]),np.mean(AUC[i]),np.std(AUC[i])))        