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
from model_omni_new import Siamese
import time
import numpy as np
import gflags
import sys
#from collections import deque
import os
import math
from scipy.special import softmax, comb, perm
from sklearn import mixture
from sklearn.metrics import roc_auc_score
from itertools import combinations, permutations
from PIL import Image

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


if __name__ == '__main__':

    Flags = gflags.FLAGS
    gflags.DEFINE_bool("cuda", True, "use cuda")
    gflags.DEFINE_integer("feature_n", 2, "how many features are chose")
    gflags.DEFINE_string("features", '0,20',  "which features are used, it can be specific feature id ('0,20') or 'full'(means using all features)")
    gflags.DEFINE_string("train_path", "../omniglot/python/images_background", "training folder")
    gflags.DEFINE_string("test_path", "../omniglot/python/images_test", 'path of testing folder')
    gflags.DEFINE_string("backbone", 'conv', "backbone model architecture")
    gflags.DEFINE_integer("way", 5, "the N ways of 'N way K shot learning'")
    gflags.DEFINE_float("dropout_p", 0.0, "positive dropout probability")  
    gflags.DEFINE_float("dropout_n", 0.0, "negative dropout probability")
    gflags.DEFINE_integer("val_way", 3, "validation based on N ways")
    gflags.DEFINE_float("hloss_alpha", 0.2, "coarse loss factor")
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
    way_list = [Flags.way,2*Flags.way,4*Flags.way, 6*Flags.way]
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
        
        
    # Using the samples of the support set to construct negative pairs
    def task_image_pairs(task_images,label,task_way):
        new_label = []
        for i, pair in enumerate(list(combinations(range(task_way), 2))):
            if i==0:
                task_images_1 = task_images[pair[0]]
                task_images_2 = task_images[pair[1]]       
            else:
                task_images_1 = torch.vstack((task_images_1,task_images[pair[0]]))
                task_images_2 = torch.vstack((task_images_2,task_images[pair[1]]))
        new_label = [0.0]*int(comb(task_way,2))
        return task_images_1, task_images_2, torch.from_numpy(np.array([new_label])).type(torch.FloatTensor),len(new_label)
        
    def evaluation(data_loader,feature_n,dropout_p,dropout_n,ways,filenames,is_hloss,backbone):
        net.eval()
        right, error = 0, 0
        soft_score = []
        prop_n_score = []
        gt = []
        pred_label = []
        for _, (test1, test2, label) in enumerate(data_loader, 1):
            # generate negative pairs according to the support set of each task.        
            img_gen1, img_gen2, label_gen, b_size = task_image_pairs(test2,label,ways)
            c = img_gen1.size(-3)
            h = img_gen1.size(-2)
            w = img_gen1.size(-1)
            img_gen1 = img_gen1.view(c, -1, h, w)
            img_gen2 = img_gen2.view(c, -1, h, w)
            if Flags.cuda:
                test1, test2 = test1.cuda(), test2.cuda()
                img_gen1, img_gen2 = img_gen1.cuda(), img_gen2.cuda()
            test1, test2 = Variable(test1), Variable(test2)
            img_gen1, img_gen2 = Variable(img_gen1), Variable(img_gen2)
            output_1, output = net.forward(test1, test2, is_hloss)
            _, out_gen = net.forward(img_gen1, img_gen2, is_hloss)
            output = output.data.cpu()

            output = output.numpy()

            out_gen = out_gen.data.cpu().numpy()
            
            #### add outlier detection#############

            dpgmm_n = mixture.BayesianGaussianMixture(n_components=b_size, tol= 5*1e-3,covariance_type='diag', weight_concentration_prior=1./b_size, max_iter=500)
            dpgmm_n.fit(out_gen)
            #obatain the probabilities of out_gen under fitted DPGMM distribution
            Neg_prob_ = dpgmm_n.score_samples(out_gen)            
            Neg_prob = dpgmm_n.score_samples(np.max(output,axis=0,keepdims=True))
            # standardization             
            Neg_prob = (np.squeeze(Neg_prob)-np.mean(Neg_prob_))/np.std(Neg_prob_)             
            gt.append(np.max(label.cpu().numpy()))
            soft_score.append(np.max(output))       
            #new open-set recognition metric                 
            prop_n_score.append(np.max(output)-1.0*Neg_prob)
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
            auc_n = roc_auc_score(np.asarray(gt,dtype=np.float32), np.asarray(prop_n_score,dtype=np.float32))
        except:
            auc = 0.0
            auc_n = 0.0

        print('*'*70)
        print('\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(right, error, right*1.0/(right+error)))
        print('*'*70)
        with open(filenames, 'a') as f:
            f.write('current_acc:'+str(acc)+'\n')
            f.write('auc_softscore:'+str(auc)+'\n')
            f.write('auc_negative_score:'+str(auc_n)+'\n')
        return acc, auc, auc_n, np.asarray(prop_n_score,dtype=np.float32), np.asarray(soft_score,dtype=np.float32), np.asarray(pred_label,dtype=np.float32), np.asarray(gt,dtype=np.float32)
        
    # multi-model ensemble: loading models
    models_list = list(np.array([-400,-300,-200,-100,0]) + int(Flags.checkbreakpoint))
    ACC = {}
    AUC = {}
    AUC_n = {}

    for n in range(len(way_list)):
        ACC[n] = []
        AUC[n] = []
        AUC_n[n] = []
   
    for i, ways in enumerate(way_list): 
        soft_score = np.empty((len(models_list),Flags.times*2))
        prop_n_score = np.empty((len(models_list),Flags.times*2))
        pred_label = np.empty((len(models_list),Flags.times))
        gt = np.empty((Flags.times))
        testSet = OmniglotTest(Flags.test_path, transform=transforms.ToTensor(),times = Flags.times*2, way = ways, features=features)
        testLoader = DataLoader(testSet, batch_size=ways, shuffle=False, num_workers=Flags.workers)
        for j, model_id in enumerate(models_list):
        #loading checkpoint
            if Flags.resume:
                print('-----------check point loading------------------')
                path_checkpoint = Flags.model_path + '/model-'+ Flags.backbone +'-dropout' + str(Flags.dropout_p) +'_' +str(Flags.dropout_n)+'-distortion-new-1.0+hloss_'  + str(Flags.hloss_alpha) +'-' + str(model_id) + ".pt"
                checkpoint = torch.load(path_checkpoint)         
                net.load_state_dict(checkpoint['model_state_dict']) 
                acc_best_single = checkpoint['acc_best_single']
                test_batch = checkpoint['batch']
                print('------------finished-----------------')
            print("test_epoch:",test_batch)
            acc, auc, auc_n, prop_n_score[j,:], soft_score[j,:], pred_label[j,:], gt = evaluation(testLoader,feature_n,Flags.dropout_p,Flags.dropout_n,ways,Flags.backbone+'_adaptive_omni_randomway_dropout' + str(Flags.dropout_p) +'_' +str(Flags.dropout_n) + '_test_results_1.0+hloss_'+ str(Flags.hloss_alpha)+'_'+ str(ways)+'way_'+ features +'_feature' + '_val'+'_based_'+str(Flags.val_way)+'way_ensemble',is_hloss=Flags.is_hloss,backbone=Flags.backbone)
            ACC[i].append(acc)
            AUC[i].append(auc)
            AUC_n[i].append(auc_n)
        # multi-model ensemble:  voting for accuracy
        acc_ensemble = (np.sum(np.sum(pred_label, axis=0)<((len(models_list)//2)+1))+0.0)/Flags.times
        # multi-model ensemble: using the mean of the decision metirc for auroc
        soft_score_mean = np.mean(soft_score, axis=0)
        prop_n_score_mean = np.mean(prop_n_score, axis=0)
        try:
            auc_ensemble = roc_auc_score(gt, soft_score_mean)
            auc_ensemble_prop_n = roc_auc_score(gt, prop_n_score_mean)
        except:
            auc_ensemble = 0.0
            auc_ensemble_prop_n = 0.0

        print(' %d way:ACC_ave: %f, ACC_std: %f, AUC_ave: %f, AUC_std: %f,AUC_n_ave: %f, AUC_n_std: %f, ACC_ensemble: %f, AUC_ensemble: %f, AUC_n_ensemble: %f'%(way_list[i],np.mean(ACC[i]),np.std(ACC[i]),np.mean(AUC[i]),np.std(AUC[i]),np.mean(AUC_n[i]),np.std(AUC_n[i]),acc_ensemble, auc_ensemble, auc_ensemble_prop_n))        