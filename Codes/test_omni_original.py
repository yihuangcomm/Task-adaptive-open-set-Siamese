import torch
import pickle
import torchvision
# from torch.optim import lr_scheduler
# import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms
from mydataset_omni import OmniglotTrain, OmniglotTest
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model_omni import Siamese
import time
import numpy as np
import gflags
import sys
#from collections import deque
import os
import math
from scipy.special import softmax
from sklearn import mixture
from sklearn.metrics import roc_auc_score

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


if __name__ == '__main__':

    Flags = gflags.FLAGS
    gflags.DEFINE_bool("cuda", True, "use cuda")
    gflags.DEFINE_integer("feature_n", 2, "how many features are chose")
    gflags.DEFINE_string("features", '0,20',  "which features are used, it can be specific feature id ('0,20') or 'full'(means using all features)")
    gflags.DEFINE_string("train_path", "../omniglot/python/images_background", "path of training folder")
    gflags.DEFINE_string("test_path", "../omniglot/python/images_test", 'path of testing folder')
    gflags.DEFINE_string("backbone", 'conv', "backbone model architecture")
    gflags.DEFINE_integer("way", 5, "the N ways of 'N way K shot learning'")
    gflags.DEFINE_float("dropout_p", 0.0, "positive dropout probability")  
    gflags.DEFINE_float("dropout_n", 0.0, "negative dropout probability")
    gflags.DEFINE_integer("val_way", 3, "validation based on N ways")
    gflags.DEFINE_float("hloss_alpha", 0.0, "coarse loss factor")
    # gflags.DEFINE_integer("shot", 5, "the K shots of 'N way K shot learning'")
    gflags.DEFINE_string("times", 600, "number of closed-set query tasks, total query tasks is 2*times including open- and closed-set tasks.")
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size", 128, "number of batch size")
    gflags.DEFINE_float("lr", 0.00006, "learning rate") 
    gflags.DEFINE_integer("show_every", 10, "show result after each show_every iter.")
    gflags.DEFINE_integer("save_every", 100, "save model after each save_every iter.")
    gflags.DEFINE_integer("test_every", 100, "test model after each test_every iter.")
    gflags.DEFINE_integer("max_iter", 50000, "number of iterations before stopping")
    gflags.DEFINE_string("model_path", "/siamese_model_omni_new", "path to store model")
    gflags.DEFINE_string("gpu_ids", "0,1,2,3", "gpu ids used to train")
    gflags.DEFINE_string('checkbreakpoint','3200', "checkpoint batch")
    gflags.DEFINE_bool('resume', True, "training from checkpoint") 
    gflags.DEFINE_bool('is_hloss', False, "whether applying hloss") 
    
    
    Flags(sys.argv)
    print(Flags.is_hloss,Flags.hloss_alpha,Flags.backbone)

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    feature_n = Flags.feature_n
    features = Flags.features
    way_list = [Flags.way,2*Flags.way,4*Flags.way,6*Flags.way]
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
        
    def evaluation(data_loader,feature_n,way_id,filenames,is_hloss):
        net.eval()
        right, error = 0, 0
        soft_score = []
        gt = []
        pred_label = []
        for _, (test1, test2, label) in enumerate(data_loader, 1):
            if Flags.cuda:
                test1, test2 = test1.cuda(), test2.cuda()
            test1, test2 = Variable(test1), Variable(test2)
            output_1, output = net.forward(test1, test2, is_hloss)
            output = output.data.cpu()
            output = output.numpy()
            gt.append(np.max(label.cpu().numpy()))
            soft_score.append(np.max(output))
            pred = np.argmax(output)
            
            if np.max(label.cpu().numpy())== 1.0:
                if pred == 0:
                    right += 1
                    pred_label.append(0.0)
                else: 
                    error += 1
                    pred_label.append(1.0)

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
        return acc, auc, np.asarray(soft_score,dtype=np.float32), np.asarray(pred_label,dtype=np.float32), np.asarray(gt,dtype=np.float32)

    models_list = list(np.array([-400,-300,-200,-100,0]) + int(Flags.checkbreakpoint))
    ACC = {}
    AUC = {}
    for n in range(len(way_list)):
        ACC[n] = []
        AUC[n] = []
    
    for i, ways in enumerate(way_list): 
        soft_score = np.empty((len(models_list),Flags.times*2))
        pred_label = np.empty((len(models_list),Flags.times))
        gt = np.empty((Flags.times))
        testSet = OmniglotTest(Flags.test_path, transform=transforms.ToTensor(),times = Flags.times*2, way = ways, features=features)
        testLoader = DataLoader(testSet, batch_size=ways, shuffle=False, num_workers=Flags.workers)
        for j, model_id in enumerate(models_list):
        #loading checkpoint
            if Flags.resume:
                print('-----------check point loading------------------')
                path_checkpoint = Flags.model_path + '/model-'+ Flags.backbone +'-dropout'+ str(Flags.dropout_p) +'_' +str(Flags.dropout_n)+'-distortion-new-1.0+hloss_'  + str(Flags.hloss_alpha) +'-' + str(model_id) + ".pt"
                checkpoint = torch.load(path_checkpoint)         
                net.load_state_dict(checkpoint['model_state_dict']) 
                acc_best_single = checkpoint['acc_best_single']
                test_batch = checkpoint['batch']
                print('------------finished-----------------')
            print("test_epoch:",test_batch)
            acc, auc, soft_score[j,:], pred_label[j,:], gt = evaluation(testLoader,feature_n,i,Flags.backbone+'_omni_randomway_dropout' + str(Flags.dropout_p) +'_' +str(Flags.dropout_n) + '_test_results_1.0+hloss_'+ str(Flags.hloss_alpha)+'_'+ str(ways)+'way_'+ features +'_feature' + '_val'+'_based_'+str(Flags.val_way)+'way_ensemble',is_hloss=Flags.is_hloss)
            ACC[i].append(acc)
            AUC[i].append(auc)
        acc_ensemble = (np.sum(np.sum(pred_label, axis=0)<((len(models_list)//2)+1))+0.0)/Flags.times
        soft_score_mean = np.mean(soft_score, axis=0)
        try:
            auc_ensemble = roc_auc_score(gt, soft_score_mean)
        except:
            auc_ensemble = 0.0

    # for i in range(len(way_list)):
        print(' %d way:ACC_ave: %f, ACC_std: %f, AUC_ave: %f, AUC_std: %f, ACC_ensemble: %f, AUC_ensemble: %f'%(way_list[i],np.mean(ACC[i]),np.std(ACC[i]),np.mean(AUC[i]),np.std(AUC[i]),acc_ensemble, auc_ensemble))        
