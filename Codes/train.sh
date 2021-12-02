#!/bin/bash


python3 train_network.py --train_path ../train_train --test_path ../val --gpu_ids 0  --model_path ../siamese_model_40_10_10_new  --dropout_p 0.4 --dropout_n 0.6 --hloss_alpha 0.3 --feature_n 2 --features 0,20

#python3 train_network_original.py --train_path ../train_train --test_path ../val --gpu_ids 0  --model_path ../siamese_model_40_10_10_new  --dropout_p 0.0 --dropout_n 0.0 --hloss_alpha 0.0 --feature_n 2 --features 0,20

#python3 train_omni.py --train_path ../omniglot/python/images_background --test_path ../omniglot/python/images_validation --backbone conv --gpu_ids 0 --dropout_p 0.0 --dropout_n 0.0 --hloss_alpha 0.3 --model_path ../siamese_model_omni_new --is_hloss True 




