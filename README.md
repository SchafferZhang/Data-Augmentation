# Data-Augmentation

This is the basic implementaion of classificaiton network without any traditional data augmentaion technique. And you can also achieve the naive implementation of GAN data augmentaion, in which the G takes in latent `z` as well as the class label `c` and outputs a fake image, while the D try to differentiate the real image from the fake. What different is the D outputs `2N` probabilities indicating the classes the input belongs to. 

## Usage

### Run classification without any data augmentaion techniques

`python dcgan.py --mode train_classifier`

### Run GAN data augmentaion network

`python dcgan.py --mode train`

### To generate samples, run

`python dcgan.py --mode generate --nice`
