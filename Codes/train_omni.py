import torch
import pickle
import torchvision
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
import os
from collections import deque
import math
from scipy.special import softmax
from sklearn import mixture
from sklearn.metrics import roc_auc_score

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


if __name__ == '__main__':

    Flags = gflags.FLAGS
    gflags.DEFINE_bool("cuda", True, "use cuda")
    gflags.DEFINE_integer("feature_n", 2, "how many features are used")
    gflags.DEFINE_string("features", '0,20', "which features are used, it can be specific feature id ('0,20') or 'full'(means using all features)")
    gflags.DEFINE_string("train_path", "../omniglot/python/images_background", "path of training folder")
    gflags.DEFINE_string("test_path", "../omniglot/python/images_validation", 'path of validation/testing folder')
    gflags.DEFINE_string("backbone","conv", "backbone of model architecture")
    gflags.DEFINE_integer("way", 5, "the N ways of 'N-way K-shot learning'")
    gflags.DEFINE_float("dropout_p", 0.0, "positive dropout probability")  
    gflags.DEFINE_float("dropout_n", 0.0, "negative dropout probability")
    gflags.DEFINE_float("hloss_alpha", 0.3, "coarse loss factor") 
    gflags.DEFINE_string("times", 400, "number of closed-set query tasks, total query tasks is 2*times including open- and closed-set tasks.")
    gflags.DEFINE_integer("workers", 1, "number of dataLoader workers.")
    gflags.DEFINE_integer("batch_size", 128, "number of batch size") 
    gflags.DEFINE_float("lr", 0.00006, "learning rate")
    gflags.DEFINE_integer("show_every", 10, "show result every 'show_every' batches.")
    gflags.DEFINE_integer("save_every", 100, "save model every 'save_every' batches.")
    gflags.DEFINE_integer("test_every", 100, "test model every 'test_every' batches.")
    gflags.DEFINE_integer("max_iter", 50000, "number of batches before stopping")
    gflags.DEFINE_string("model_path", "/siamese_model_omni_new", "path to store model")
    gflags.DEFINE_string("gpu_ids", "0", "gpu ids used to train")
    gflags.DEFINE_string('checkbreakpoint','26200', "checkpoint path")
    gflags.DEFINE_bool('resume', False, "training from checkpoint") 
    gflags.DEFINE_bool('is_hloss', True, "whether applying hloss") 
    

    Flags(sys.argv)

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])


    feature_n = Flags.feature_n
    features = Flags.features
    os.environ["CUDA_VISIBLE_DEVICES"] = Flags.gpu_ids
    print("use gpu:", Flags.gpu_ids, "to train.")
    # data loading
    trainSet = OmniglotTrain(Flags.train_path, transform=data_transforms, dropout_p=Flags.dropout_p,dropout_n=Flags.dropout_n, features=features)    
    trainLoader = DataLoader(trainSet, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)

    way_list = [Flags.way, 2*Flags.way, 4*Flags.way] # # for different support classes.
    
    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
     
    net = Siamese(feature_n)

    # multi gpu
    if len(Flags.gpu_ids.split(",")) > 1:
        net = torch.nn.DataParallel(net)

    if Flags.cuda:
        net.cuda()

    # Adam optimizer and fixed learning rate
    optimizer = torch.optim.Adam(net.parameters(),lr = Flags.lr )
    
    #create dictionary or list to store loss values and accuracies for each kind of support set.    
    train_loss = []
    test_loss = {}
    loss_val_all = 0.0
    loss_val_ave = 0.0
    loss_test = [0.]*len(way_list)
    acc_list = {}
    queue = {}
    idx = 40
    for i in range(len(way_list)):
        # train_loss[i] = []
        test_loss[i] = []
        queue[i] = deque(maxlen=5) # store 5 successive test results
        acc_list[i] = []
        
    auc_softscore = [0.0]*len(way_list)
    acc_best = [0.0]*len(way_list)
    best_batch_id = [0]*len(way_list)
    acc_best_single = [0.0]*len(way_list)
    best_single_batch_id = [0]*len(way_list)
    N = 0

    net.train()  

    optimizer.zero_grad()
  
    time_start = time.time()

    def evaluation(data_loader,feature_n,way_id,filenames,is_hloss,backbone):
        net.eval()
        right, error = 0, 0
        correct_max = []
        error_max = []
        correct_scoremax = []
        error_scoremax = []
        soft_pred = []
        soft_score = []
        gt = []
        for _, (test1, test2, label) in enumerate(data_loader, 1):
            if Flags.cuda:
                test1, test2 = test1.cuda(), test2.cuda()
            test1, test2 = Variable(test1), Variable(test2)
            output_1, output = net.forward(test1, test2, is_hloss,backbone)
            output_1, output = output_1.data.cpu(), output.data.cpu(),
            loss_t = loss_fn(output, label)            
            loss_test[way_id] += loss_t.item()
            output = output.numpy()
            gt.append(np.max(label.cpu().numpy()))
            soft_pred.append(np.max(softmax(output)))
            soft_score.append(np.max(output))
            pred = np.argmax(output)
            if np.max(label.cpu().numpy())== 1.0:
                if pred == 0:
                    right += 1
                    correct_max.append(np.max(softmax(output)))
                    correct_scoremax.append(np.max(output))
                else: 
                    error += 1
                    error_max.append(np.max(softmax(output)))
                    error_scoremax.append(np.max(output))
        acc = right*1.0/(right+error)
        acc_list[way_id].append(acc)
        try:
            auc_softscore[way_id] = roc_auc_score(np.asarray(gt,dtype=np.float32), np.asarray(soft_score,dtype=np.float32))
        except:
            auc_softscore[way_id] = 0.0
        if acc>acc_best_single[way_id]:
            acc_best_single[way_id] = acc
            best_single_batch_id[way_id] = batch_id
           
        queue[way_id].append(acc)
        
        acc_avg = 0.0
        i = 0
        for d in queue[way_id]:
            acc_avg += d
            i += 1
        acc_avg /= i
        
        if acc_avg>acc_best[way_id]:
            acc_best[way_id] = acc_avg
            best_batch_id[way_id] = batch_id
        print('*'*70)
        print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
        print('*'*70)
        with open(filenames, 'a') as f:
            f.write('current_acc:'+str(acc)+'\n')
            f.write('best_average_acc:'+str(acc_best[way_id])+'\n')
            f.write('best_acc:'+str(acc_best_single[way_id])+'\n')
            f.write('best_single_batch:'+str(best_single_batch_id[way_id])+'\n')
            f.write('best_batch:'+str(best_batch_id[way_id])+'\n')
            f.write('latest_5_sucsessive_acc:'+str(queue[way_id])+'\n')
            f.write('current_auc_softscore:'+str(auc_softscore[way_id])+'\n')
            f.write('batch_id: '+str(batch_id)+'\n')
        test_loss[way_id].append(loss_test[way_id])
        return 
    
    for batch_id, (img1, img2, label, label_1) in enumerate(trainLoader, 1):
        if batch_id > Flags.max_iter:
            break
        if Flags.cuda:
            img1, img2, label, label_1 = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda()), Variable(label_1.cuda())
        else:
            img1, img2, label, label_1 = Variable(img1), Variable(img2), Variable(label), Variable(label_1)
        optimizer.zero_grad()
        output_1, output = net.forward(img1, img2, Flags.is_hloss, Flags.backbone)
        loss = loss_fn(output, label) + Flags.hloss_alpha*loss_fn(output_1, label_1)
        loss_val_all += loss.item()
        loss_val_ave = loss_val_all/(batch_id+1.0)
        loss.backward()
        optimizer.step()
        
        if batch_id % Flags.show_every == 0 :
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s'%(batch_id, loss_val_ave, time.time() - time_start))
            time_start = time.time()
        if batch_id % Flags.save_every == 0:
        #check point
            checkpoint_dict = {'batch': batch_id, 
                   'model_state_dict': net.state_dict(), 
                   'optim_state_dict': optimizer.state_dict(),
                   'loss_val_all':loss_val_all,
                   'acc_list': acc_list,
                   'acc_best_single':acc_best_single}
            torch.save(checkpoint_dict, Flags.model_path + '/model-'+ Flags.backbone +'-dropout' + str(Flags.dropout_p) +'_' +str(Flags.dropout_n)+'-distortion-new-1.0+hloss_' + str(Flags.hloss_alpha) +'-' + str(batch_id) + ".pt")

        if batch_id % Flags.test_every == 0:
            i = 0
            for way_id in way_list:
                testSet = OmniglotTest(Flags.test_path, transform=transforms.ToTensor(), times=Flags.times*2, way=way_id, features=features)
                testLoader = DataLoader(testSet, batch_size=way_id, shuffle=False, num_workers=Flags.workers)                
                evaluation(testLoader,feature_n,i,'train_' + Flags.backbone +'_dropout_distortion_'+ str(Flags.dropout_p) +'_' +str(Flags.dropout_n)+'_results_1.0+hloss_'+ str(Flags.hloss_alpha) +'_'+str(way_id)+'way_', is_hloss=Flags.is_hloss, backbone=Flags.backbone)
                i= i+1
        train_loss.append(loss_val_ave)     

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)
    with open('test_loss', 'wb') as f:
        pickle.dump(test_loss, f)

