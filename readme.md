# ClustNet

<figure>
  <img src="/readmeFiles/clustNetScheme.png" width="100%">
  <figcaption style="
    margin: 10px 0 0 0;
    font-weight: bold;
    text-align: center;"> The framework used to train the model </figcaption>
</figure>

The aim of the project is to train a model to classify MNIST images with only a weakly supervised training. Here, weakly means that during training, the net will only know which image represent a number among 0,1,2,3,4 or a number among 5,6,7,8,9. Therefore, during training, the model is trained with binary targets. When an image is from the group (0,1,2,3,4) and from the group (5,6,7,8,9), we say it belongs respectively to the positive (P) and negative (N) class.

The model proposed here is a neural architecture of the mixture of experts family. The gate is a CNN called the clustering network (CN) and the experts are called the detecting networks (DNs).

For a 10 classes problem, there is one CN and 10/2 = 5 DNs. The CN reads the image and predicts which DN to use with a one-hot vector of size 5. Then, the selected DN predicts which of the two classes (P or N) the input image belong.

The CN and the DNs are conjointly trained to predict correctly the group using binary cross entropy.

The CN produces hard-decision : it selects only one DN and not a linear combination of the 5 DNs. To implement this, the CN outputs a multinomial distribution over the 5 DNs. During training, we sample from this distribution to obtain an index (between 1 and 5) which indicates the DN to use. During evaluation, the DN used is the one which index has the highest probability. The backpropagation is performed using the straight through (ST) estimator.

To improve training stability, the CN and the DNs share their features. In the begining of this project the CN and the DNs were CNNs and they all had separate parameters. Now they share their convolutional layers. Now, the model can be seen as follow : a convolutional encoder and then to the CN which is composed of two dense layers. The DNs read directly the representation produced by the encoder instead of using their own encoder

Two things can happen from here :

- If the capacity of the DNs is high, the gate net is going to use always the same detecting net to
separate the images.

- **If their capacity is low**, to use only one detecting net is not sufficient to solve the task. Therefore, the gate net will have to use specific experts
for specific images.
In this case, we will observe **a good correlation between the class of the image and the expert choosen
by the gate net**.

## Evaluation

To evaluate the models, I compute the clustering accuracy when the input is from class P and when input is from class N.

When input is from class P, it can be one of these 5 MNIST classes : 0,1,2,3 or 4. For each image, the CN produce a score for each DN and we consider the DN with the higher score as the DN predicted by the CN. Using the Hungarian algorithm, we obtain a bijection between the 5 DNs and the five MNIST classes and we can compute accuracy as is usually done in a fully supervised setting. The same thing is done with the class N, so we obtain two clustering accuracy. We also compute the accuracy at separating the class P and N.

## Results

We evaluate our method on MNIST and CIFAR10. We plot the three accuracies computed on the test set during training.

### MNIST

Here I first vary the batch size and the learning rate. In the figure below, you can see that with small batch size, the CN yields a performance that is much better than random.

<figure>
  <img src="/readmeFiles/linDet_testError.png" width="100%">
  <figcaption style="
    margin: 10px 0 0 0;
    font-weight: bold;
    text-align: center;"> Performance of several models on MNIST. Each color represents a model with its own batch size and learning rate value. Dashed, straight and dotted lines
                          represents binary accuracy (class P or N), positive clustering accuracy (class 0,1,2,3 or 4) and negative clustering accuracy (5,6,7,8 or 9). One can see that the models with the smaller batch size yield a good performance. </figcaption>
</figure>

Below, a model with batch size 4 and learning rate 0.01 is trained several times from scratch using different seeds. One can see the performance has high variance.

<figure>
  <img src="/readmeFiles/linDet2_testError.png" width="100%">
  <figcaption style="
    margin: 10px 0 0 0;
    font-weight: bold;
    text-align: center;"> Performance of several models on MNIST using the same learning rate and batch size but different random seed. </figcaption>
</figure>

### CIFAR10

Here I first vary the batch size and the learning rate, as in the previous experiment on MNIST. In the figure below, you can see that with small batch size, the CN yields a performance that is much better than random.

<figure>
  <img src="/readmeFiles/linDet_cifar_testError.png" width="100%">
  <figcaption style="
    margin: 10px 0 0 0;
    font-weight: bold;
    text-align: center;"> Performance of several models on CIFAR10. Each color represents a model with its own batch size and learning rate value. Dashed, straight and dotted lines
                          represents binary accuracy (class P or N), positive clustering accuracy (class 0,1,2,3 or 4) and negative clustering accuracy (5,6,7,8 or 9). One can see that the models with the smaller batch size yield a good performance. </figcaption>
</figure>



<figure>
  <img src="/readmeFiles/linDet2_cifar_testError.png" width="100%">
  <figcaption style="
    margin: 10px 0 0 0;
    font-weight: bold;
    text-align: center;"> Performance of several models on MNIST using the same learning rate and batch size but different random seed. </figcaption>
</figure>


## Installation

Clone this git first.

Then install the requirements with the requirement file located at the root of the project :

```
pip install -r requirements.txt
```

## Training and testing

To start training a CN just go in the code folder at project root and run :
```
python3 trainVal.py -c clust.config --exp_id experience1
```

The ```-c clust.config``` indicates the config file loading the default arguments. ```--exp_id experience1``` indicates the name of the experiment. The folders containing the results will have that name. This command start training a CN and write files in several folders at project root :

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

The point of this way of doing is to be able to easily test different arguments values. For example, train 4 nets with a different learning rate :

```
for lr in (0.01 0.05 0.001 0.0005)
do
	python3 trainVal.py -c clust.config --exp_id lr_values --lr $lr
done
```

The results of these four nets will all be in the same folder (here, the folder "lr_values" in "nets", "results" and "vis").

Now, let's try several number of hidden layers for the CN:

```
for nb_lay in (2 3 4 5)
do
	python3 trainVal.py -c clust.config --exp_id nb_layers --clnblayers $nb_lay
done
```

## Some visualisations


Now you want to visualize the results of the experience ```nb_layers```. The script 'processResults.py' contains several functions for that. Here's how to use each of them :

- To plot distance between the weights and their value at preceding epoch, for each epoch :
```
python3 processResults.py --exp_id nb_layers --weig
```
- To plot the mean sparsity of activation. The values after "--spar" is the list of arguments which are varying across the different nets. In this experiment, the parameters which has changed is the number of layers in the CN (i.e. ```clnblayers```). So you should type :
```
python3 processResults.py --exp_id nb_layers --spar clnblayers
```
- To plot the error, accuracy and clustering score across the training epochs for all nets in the experiment. Also plot the total variation of these curves. Again, the values after ```--acc_evol``` is the list of parameters which are varying among the different nets. So you should type :
```
python3 processResults.py --exp_id nb_layers --acc_evol clnblayers
```
- To write in a folder the test images that are misclassified by one net nets at one epoch. The first parameter after this argument is the net index and the second one is the epoch number. To study the errors of network 2 at epoch 5, type :
```
python3 processResults.py --exp_id nb_layers --failurecases 2 5
```
- For each image, sort the values at the output of the CN, which gives one vector with sorted value for each image. Compute the mean vector all images and plot that vector as a cumulated histogram. Repeat this for all epochs. This plot shows the evolution of certainty with which the CN takes a decision. This produces one image for each net.
```
python3 processResults.py --exp_id nb_layers --cludis
```
- For each detecting network, plot the feature map average variance across epochs. Positive and negative images are separated. The first argument is the net id and the second is the layer number. To study the variance of the feature maps of the network 2 at epoch 5 type :
```
python3 processResults.py --exp_id nb_layers --feat_map_var 2 5
```

- To plot t-sne on the representations produced by an encoder. The fisrt argument must be the path to an convolutional encoder (CAE) weight file and the second is the number of samples to use. If you have trained a CAE in an experiment called "encoder", which id is 1, and you want its weights at epoch 5, type :
```
python3 processResults.py --exp_id nb_layers --tsne ../nets/encoder/encoder/cae1_epoch5 1000
```

- Finally, to do those computations on the train set, just add ```--train``` to your command line.
