#!/bin/bash

python3 test_network_dmm.py --train_path ../train_train --test_path ../test --gpu_ids 0  --model_path ../siamese_model_40_10_10_new --checkbreakpoint 16600 --val_way 5 --feature_n 2 --features 0,20 --dropout_p 0.4 --dropout_n 0.6 --hloss_alpha 0.3

#python3 test_network_original.py --train_path ../train_train --test_path ../test --gpu_ids 0 --model_path ../siamese_model_40_10_10_new --checkbreakpoint 12500 --val_way 3 --feature_n 2 --features 0,20

#python3 test_omni_dmm.py --train_path ../omniglot/python/images_background --test_path ../omniglot/python/images_test --backbone conv --gpu_ids 0 --dropout_p 0.0 --dropout_n 0.0 --hloss_alpha 0.3 --model_path ../siamese_model_omni_new --checkbreakpoint 48700 --val_way 10 --is_hloss True 

#python3 test_omni_original.py --train_path ../omniglot/python/images_background --test_path ../omniglot/python/images_test --backbone conv4 --gpu_ids 0 --dropout_p 0.0 --dropout_n 0.0 --hloss_alpha 0.0 --model_path ../siamese_model_omni_new --checkbreakpoint 43200 --val_way 10 --is_hloss False 
