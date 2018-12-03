# ClustNet

The aim of the project is to train a model to classify MNIST images with only a weakly supervised training. Here, weakly means that during training, the net will only know which image represent a number among 0,1,2,3,4 or a number among 5,6,7,8,9. Therefore, during training, the model is trained with binary targets. During test the model is evaluated with the 10-class target.

The model proposed here a neural architecture of the mixture of experts family. The gate is a CNN called the clustering network (CN) and the experts are called the detecting networks (DNs).

The input is passed to the CN which decides which DN to use to
process the images. More precisely, the CN outputs a vector of weights.

The input is then passed to the DNs and their prediction is summed with weights produced by the CN.

Two things can happen from here :

- If the capacity of the DNs is high, the gate net is going to use always the same detecting net to
separate the images.

- **If their capacity is low**, to use only one detecting net is not sufficient to solve the task. Therefore, the gate net will have to use specific experts
for specific images.
In this case, we will observe **a good correlation between the class of the image and the expert choosen
by the gate net**.

## Installation

Clone this git first.

Then install the requirement file located at the root of the project :

```
pip install -r requirements.txt
```

## Training and testing

To start training a CN just go in the code folder at project root and run :
```
python3 trainVal.py -c clust.config --exp_id experience1
```

This will start training a CN and write files in several folders at project root :

- In "results" you will find a folder named "experience1" containing results named like : "all_scores_net1_epoch1.csv".
By looking at the file name, we see that the file contains results about the test performance of net 1 after epoch 1.
(There is also similar files for training images named like : "all_scores_net0_epoch1_train.csv")
These csv files contains for each test image :
  - The 10-class target
  - The binary target
  - The log-probability pairs produced to predict the binary target (2 numbers)
  - The output of the CN predicting which DN to use. This is also a distribution.

- In "nets" you will find (in folder "experience1") the weigths files for each net after each epoch. There is also one config file
for each net.

- In "vis" you will find (in folder "experience1") visualisations of the performance of each net. These include image with name like "perm_det_2_1.png". From the name of this file we see that it's about the net number 2 and the epoch number 1. It represents the confusion matrix between the classes [0,1,2,3,4]. There also is a confusion matrix for the classes [5,6,7,8,9]

## Some visualisations
