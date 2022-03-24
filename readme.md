## Task Adaptive Siamese Neural Networks for Open-Set Recognition of Encrypted Network Traffic With Bidirectional Dropout

An implementation of this paper in pytorch on the encrypted network trafffic dataset and the Omniglot dataset.

## requirement

Experiments were implemented on the high performance computing (HPC) Platform of a University. 

Experiments enviroments settings are as below:

Software:

- Python 3.8.8 
- torch 1.8.1
- torchvision 0.9.1
- opencv-python 4.5.3.56
- scipy 1.6.2
- sklearn 0.24.1
- pandas 1.2.4
- numpy 1.20.1
- collection 0.1.6
- PIL (pillow) 8.2.0
- python-gflags 3.1.2


- OS: Red Hat Enterprise Linux Server release 7.9 

Hardware:

- GPU number: 1
- GPU type: NVIDIA P100 
- GPU RAM: 12GB

- CPU number: 1
- CPU type: AMD64
- CPU RAM: 4000MB



## Run step

train and test by running .sh files.

```
sh train.sh
sh test.sh

```
## Reference
Fang Pin, Siamese-pytorch, (2019), GitHub repository, https://github.com/fangpin/siamese-pytorch 

