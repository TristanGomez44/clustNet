import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from args import str2bool
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
import numpy as np

class oneHotActivation(torch.autograd.Function):
    """Sample a batch of one-hot vectors from a batch of discrete distribution given as input

    Takes a batch of discrete distribution (i.e. a list of vectors) as input, each value in a distribution representing
    the probability of sampling one class. The one-hot vector returned has its one at the i-th position,
    where i is the index of the sampled class.

    The backward pass is estimated using the straight-trough method (cf. "Estimating or
    Propagating Gradients Through Stochastic Neurons for Conditional Computation" by Bengio,
    Léonard and Courville).
    """

    @staticmethod
    def forward(ctx, input):
        '''Sample a batch of one-hot vector from a batch of discrete distribution

        Args:
            ctx (object): the object used the store the input for the backward pass
            input (numpy.ndarray): the batch of discrete distribution from which the one hot vector are sampled.
            Shape must be (batch_size,number_of_classes).
        Returns:
            oneHot: A batch of one hot vector sampled from the distribution. Shape is (batch_size,number_of_classes).
        '''

        ctx.save_for_backward(input)

        #Sampling a class according to distribution
        choosenIndexs = Categorical(input).sample()  # equal probability of 0, 1, 2, 3

        #Building a one-hot vector
        oneHot = torch.zeros(input.size())

        if input.is_cuda:
            oneHot = oneHot.cuda()

        batchInds = np.arange(input.size(0))
        oneHot[batchInds,choosenIndexs] = 1

        return oneHot


    @staticmethod
    def backward(ctx, grad_output):
        '''Estimate the backward pass using the straight-through estimator
        Only used by pytorch during backward pass.
        '''

        #The gradient is estimated with the straight-through method
        input = ctx.saved_tensors

        return grad_output

class oneHotActivationDeterministic(torch.autograd.Function):
    """Compute a batch of one-hot vector from a batch of discrete distribution given as input

    Takes a batch of discrete distribution (i.e. a list of vectors) as input and returns a batch of one-hot
    vectors where the one are located at the index of the most probable class

    """
    @staticmethod
    def forward(ctx, input):
        '''Compute a batch of one-hot vector from a batch of discrete distribution. (i.e. a list of vectors)
        The ones are positioned at the index of the most probable class.

        Args:
            ctx (object): the object used the store the input for the backward pass
            input (numpy.ndarray): the batch of discrete distribution from which the one hot vector are computed.
            Shape must be (batch_size,number_of_classes).
        Returns:
            oneHot: A batch of one hot vector computed from the distribution. The ones are positioned at the index
            of the most probable class. Shape is (batch_size,number_of_classes).
        '''

        ctx.save_for_backward(input)
        oneHot = torch.zeros(input.size())

        if input.is_cuda:
            oneHot = oneHot.cuda()

        _,choosenIndexs = torch.max(input,dim=1)

        batchInds = np.arange(input.size(0))
        oneHot[batchInds,choosenIndexs] = 1

        return oneHot

    @staticmethod
    def backward(ctx, grad_output):
        '''Estimate the backward pass using the straight-through estimator
        Only used by pytorch during backward pass.
        '''

        #The gradient is estimated with the straight-through method
        input = ctx.saved_tensors

        return grad_output

class BasicConv2d(nn.Module):
    """A basic 2D convolution layer

    This layer integrates 2D batch normalisation and relu activation

    Comes mainly from torchvision code :
    https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
    Consulted : 19/11/2018

    """
    def __init__(self, in_channels, out_channels,use_bn=True, **kwargs):

        '''
        Args:
            in_channels (int): the number of input channel
            out_channels (int): the number of output channel
            use_bn (boolean): whether or not to use 2D-batch normalisation
             **kwargs: other argument passed to the nn.Conv2D constructor
        '''

        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not(use_bn), **kwargs)
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)

        return F.relu(x, inplace=True)

class BasicTranposedConv2d(BasicConv2d):
    ''' Implement transposed 2D convolution with 2D batch norm and relu activation '''

    def  __init__(self, in_channels, out_channels, use_bn=True,**kwargs):
        super(BasicTranposedConv2d,self).__init__(in_channels, out_channels, **kwargs)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=not(use_bn), **kwargs)

class ConvFeatExtractor(nn.Module):
    """A convolutional feature extractor

    It is a stack of BasicConv2d layers with residual connections and it also comprise
    two maxpooling operation (one after layer 1 and one after the last layer)

    """

    def __init__(self,inChan,chan,nbLay,ker,maxPl1,maxPl2,applyDropout2D,outChan=None):
        '''
        Args:
            inChan (int): the number of input channel
            chan (int): the number of channel in every layer of the network
            outChan (int): the number of channel at the end layer of the network. Default value is the number of channel\
                in the other layers.
            avPool (boolean): whether of not to use average pooling
            nbLay (int): the number of layer of the net
            ker (int): the size of side of the kernel to use (the kernel is square)
            maxPl1 (int): the max-pooling size after the first convolutional layer
            maxPl2 (int): the max-pooling size after the before-to-last convolutional layer
            applyDropout2D (boolean): whether or not to use 2D-dropout on the middle layers during training
        '''
        super(ConvFeatExtractor,self).__init__()

        if outChan is None:
            self.outChan = chan
        else:
            self.outChan = outChan

        self.nbLay = nbLay
        self.chan = chan
        self.inChan = inChan
        self.ker = ker

        self.maxPl1 = maxPl1
        self.maxPl2 = maxPl2

        self.poollLay1 = nn.MaxPool2d(maxPl1,return_indices=True)
        self.poollLay2 = nn.MaxPool2d(maxPl2,return_indices=True)

        self.applyDropout2D = applyDropout2D

        self.convs = nn.ModuleList([BasicConv2d(inChan, chan,kernel_size=self.ker)])

        self.padd = nn.ZeroPad2d(self.computePaddingSize(self.ker))

        if self.nbLay > 2:

            #Padding is applied on every layer so that the feature map size stays the same at every layer
            self.convs.extend([BasicConv2d(chan,chan,kernel_size=self.ker) for i in range(self.nbLay-2)])

        self.convs.append(BasicConv2d(chan, outChan,kernel_size=self.ker))

        self.drop2DLayer = nn.Dropout2d()

    def forward(self,x):
        ''' Compute the forward pass of the stacked layer
        Returns:
            x (torch.autograd.variable.Variable): the processed batch
            actArr (list) the list of activations array of each layer
            maxPoolInds (dict): a dictionnary containing two objects : the indexs of the maximum elements for the first maxpooling
                and the second maxpooling. These two objects are obtained by the return of the nn.MaxPool2d() function
        '''

        actArr = []
        netshape ="in : "+str(x.size())+"\n"
        maxPoolInds = {}

        for i, l in enumerate(self.convs):
            if  i != 0 and self.applyDropout2D:
                x = self.drop2DLayer(x)

            #Compute next layer activations
            if (i != len(self.convs)-1 or self.chan==self.outChan) and (i != 0 or self.chan==self.inChan):
                #Add residual connection for all layer except the first and last (because channel numbers need to match)
                x = self.padd(l(x))+x
            else:
                x = self.padd(l(x))

            actArr.append(x)
            netshape +="conv : "+str(x.size())+"\n"

            if i == len(self.convs)//2-1:
                x,inds1 = self.poollLay1(x)
                maxPoolInds.update({"inds1":inds1})
            elif i == len(self.convs)-1:

                x,inds2 = self.poollLay2(x)
                maxPoolInds.update({"inds2":inds2})

        actArr.append(x)
        return x,actArr,maxPoolInds

    def computePaddingSize(self,kerSize):
        ''' Compute the padding size necessary to compensate the size reduction induced by a conv operation
        Args:
            kerSize (int): the size of the kernel (assumed squarred)
        '''

        halfPadd = (kerSize-1)//2

        if kerSize%2==0:
            padd = (halfPadd,1+halfPadd,halfPadd,1+halfPadd)
        else:
            padd = (halfPadd,halfPadd,halfPadd,halfPadd)

        return padd

class ConvDecoder(ConvFeatExtractor):
    """A convolutional feature decoder. It is the symmetric of ConvFeatExtractor"""

    def __init__(self,inChan,chan,nbLay,ker,maxPl1,maxPl2,applyDropout2D,outChan=None):
        '''
        Args:
            inChan (int): the number of channel of the representation to decode
            chan (int): the number of channel in every layer of the network
            nbLay (int): the number of layer of the net
            ker (int): the size of side of the kernel to use (the kernel is square)
            maxPl1 (int): the max-pooling size after the first convolutional layer
            maxPl2 (int): the max-pooling size after the before-to-last convolutional layer
            applyDropout2D (boolean): whether or not to use 2D-dropout on the middle layers during training
        '''
        super(ConvDecoder,self).__init__(inChan,chan,nbLay,ker,maxPl1,maxPl2,applyDropout2D,outChan=outChan)

        self.poollLay1 = nn.MaxUnpool2d(maxPl1)
        self.poollLay2 = nn.MaxUnpool2d(maxPl2)

        self.convs = nn.ModuleList([BasicTranposedConv2d(inChan, chan,kernel_size=self.ker)])

        if self.nbLay > 2:
            #Padding is applied on every layer so that the feature map size stays the same at every layer
            self.convs.extend([BasicTranposedConv2d(chan,chan,kernel_size=self.ker) for i in range(self.nbLay-2)])

        self.convs.append(BasicTranposedConv2d(chan, outChan,kernel_size=self.ker))

        self.unpTup = self.computePaddingSize(self.ker)

    def unpadd(self,x):
        ''' Remove the cells added by padding '''
        return x[:,:,self.unpTup[0]:x.size(2)-self.unpTup[1],self.unpTup[2]:x.size(3)-self.unpTup[3]]

    def forward(self,x,maxPoolInds):
        ''' Returns the output of the stacked transposed conv layers
        Args:
            x (torch.autograd.variable.Variable): the batch of representations to decode.
            maxPoolInds (dict): a dictionnary returned by the forward() method of ConvFeatExtractor
        Returns:
            x (torch.autograd.variable.Variable): the processed batch
            actArr (the list of activations array of each layer)
        '''

        actArr = []
        netshape ="in : "+str(x.size())+"\n"

        for i, l in enumerate(self.convs):

            if  i != 0 and self.applyDropout2D:
                x = self.drop2DLayer(x)

            if i == len(self.convs)//2:
                x = F.pad(x, self.computeUnpoolPaddingSize(x.size()[2:],maxPoolInds["inds1"].size()[2:]), "constant", 0)
                x = self.poollLay1(x,maxPoolInds["inds1"])

            elif i == 0:
                x = F.pad(x, self.computeUnpoolPaddingSize(x.size()[2:],maxPoolInds["inds2"].size()[2:]), "constant", 0)
                x = self.poollLay2(x,maxPoolInds["inds2"])

            if (i != 0 or self.chan==self.inChan) and (i != len(self.convs)-1 or self.chan==self.outChan):
                x = self.unpadd(l(x))+x
            else:
                x = self.unpadd(l(x))

            actArr.append(x)
            netshape +="conv : "+str(x.size())+"\n"

        actArr.append(x)

        return x,actArr

    def computeUnpoolPaddingSize(self,imgSize,unpaddedImgSize):
        ''' Takes the sizes of an image and a padded one and return the padding applied
        Args:
            imgSize (tuple): the size of the padded image
            unpaddedImgSize (tuple): the size of the unpadded image
        Returns:
            padd (tuple): a 4-tuple indicating how many cells have been padded on left, right, top and bottom respectively
        '''

        heigthPadd = unpaddedImgSize[0] - imgSize[0]
        widthPadd = unpaddedImgSize[1] - imgSize[1]

        if heigthPadd%2 == 0:
            padd = (heigthPadd//2,heigthPadd//2)
        else:
            padd = (heigthPadd//2+1,heigthPadd//2)

        if widthPadd%2 == 0:
            padd = (padd[0],padd[1],widthPadd//2,widthPadd//2)
        else:
            padd = (padd[0],padd[1],widthPadd//2+1,widthPadd//2)

        return padd

class CAE(nn.Module):
    """A Convolutional Autoencoder  module

    The weight of the encoder can be used to initialize the weights of a clustering network

    """

    def __init__(self,inSize,inChan,chan,hidd_repr_size,nbLay,ker,maxPl1,maxPl2,applyDropout2D):
        """
        Args:
            check ConvFeatExtractor module constructor
        """

        super(CAE,self).__init__()

        self.convFeat = ConvFeatExtractor(inChan=inChan,chan=chan,outChan=chan,\
                                          nbLay=nbLay,ker=ker,maxPl1=maxPl1,maxPl2=maxPl2,applyDropout2D=applyDropout2D)

        convSize =(inSize//(maxPl1*maxPl2))

        self.dense_enc = nn.Linear(chan*convSize*convSize,int(hidd_repr_size))
        self.dense_dec = nn.Linear(int(hidd_repr_size),chan*convSize*convSize)

        self.convDec = ConvDecoder(inChan=chan,chan=chan,outChan=inChan,\
                                          nbLay=nbLay,ker=ker,maxPl1=maxPl1,maxPl2=maxPl2,applyDropout2D=applyDropout2D)

    def computeHiddRepr(self,x):
        ''' Computes the hidden representation of a batch of input '''

        x,_,inds = self.convFeat(x)

        size = x.size()
        #Flattening
        x = x.view(size[0],-1)

        hiddRepr = self.dense_enc(x)
        return hiddRepr,inds,size

    def forward(self,x):
        ''' Computes the reconstruction of a batch of input '''

        #Encoding
        hiddRepr,inds,size = self.computeHiddRepr(x)

        #Decoding
        x = self.dense_dec(hiddRepr)

        #Reshaping in NCHW
        x = x.view(size)

        x,_ = self.convDec(x,inds)
        return x

class CNN(nn.Module):
    """A CNN module"""

    def __init__(self,inSize,inChan,chan,avPool,nbLay,ker,maxPl1,maxPl2,applyDropout2D,nbDenseLay,sizeDenseLay,nbOut,applyLogSoftmax=True,outChan=None):
        """
        Args:
            nbOut (int): the number of output at the last dense layer
            applyLogSoftmax (bool): whether or not to apply the nn.functional.log_softmax function in the last dense layer
            other arguments : check ConvFeatExtractor module constructor
        """

        if outChan is None:
            outChan=chan

        super(CNN,self).__init__()

        self.avPool = avPool
        if nbLay != 0:
            self.convFeat = ConvFeatExtractor(inChan=inChan,chan=chan,outChan=outChan,nbLay=nbLay,\
                                              ker=ker,maxPl1=maxPl1,maxPl2=maxPl2,applyDropout2D=applyDropout2D)
        else:
            self.convFeat = None

        self.applyLogSoftmax = applyLogSoftmax

        if nbDenseLay == 1:

            if nbLay != 0:
                if not self.avPool:
                    InputConv2Size = inSize//maxPl1
                    InputLinearSize = InputConv2Size//maxPl2
                    dense = nn.Linear(InputLinearSize*InputLinearSize*outChan,nbOut)
                else:
                    dense = nn.Linear(outChan,nbOut)
            else:
                #If the input has 0 channel it indicates it is a vector and not an matrix representation of the input
                if inChan==0:
                    dense = nn.Linear(inSize,nbOut)
                else:
                    dense = nn.Linear(inSize*inSize*inChan,nbOut)

            self.denseLayers = dense

        elif nbDenseLay == 0:
            self.denseLayers = None

        else:

            if self.avPool:
                raise ValueError("Cannot use average pooling and more than one dense layer")
            print("More than one dense layer")

            if nbLay != 0:
                InputConv2Size = int((inSize-(ker-1))/maxPl1)
                InputLinearSize = int((InputConv2Size-(ker-1))/maxPl2)

                self.denseLayers = nn.ModuleList([nn.Linear(InputLinearSize*InputLinearSize*outChan,sizeDenseLay),nn.ReLU()])
            else:

                InputLinearSize = inSize
                #If the input has 0 channel it indicates it is a vector and not an matrix representation of the input

                if inChan==0:
                    self.denseLayers = nn.ModuleList([nn.Linear(InputLinearSize,sizeDenseLay),nn.ReLU()])
                else:
                    self.denseLayers = nn.ModuleList([nn.Linear(InputLinearSize*InputLinearSize*inChan,sizeDenseLay),nn.ReLU()])

            for i in range(nbDenseLay-2):
                self.denseLayers.extend([nn.Linear(sizeDenseLay,sizeDenseLay),nn.ReLU()])

            self.denseLayers.append(nn.Linear(sizeDenseLay,nbOut))
            self.denseLayers = nn.Sequential(*self.denseLayers)
    def forward(self,x):
        '''Computes the output of the CNN

        Returns:
            x (torch.autograd.variable.Variable): the batch of predictions
            actArr (the list of activations array of each layer)
        '''

        actArr = []

        if self.convFeat:
            x,actArr,netShape = self.convFeat(x)

        if self.avPool:
            x = x.sum(dim=-1,keepdim=True).sum(dim=-2,keepdim=True)

        if self.denseLayers:

            #Flattening the convolutional features
            x = x.view(x.size()[0],-1)

            x = self.denseLayers(x)
            actArr.append(x)

            if self.applyLogSoftmax:
                x = F.log_softmax(x, dim=1)

        return x, actArr

    def setWeights(self,params,cuda,noise_init):
        '''Set the weight of the extractor
        Args:
            params (dict): the dictionnary of tensors used to set the extractor parameters. This must be the parameters of a CNN module
            cuda (bool): whether or not to use cuda
            noise_init (float): the proportion of noise to add to the weights (relative to their norm). The noise is sampled from
            a Normal distribution and then multiplied by the norm of the tensor times half this coefficient. This means that
            95%% of the sampled noise will have its norm value under noise_init percent of the parameter tensor norm value

        '''
        #Used to determine if a parameter tensor has already been set.
        setKeys = []

        for key in params.keys():
            #The parameters may come from a CNN or CAE module so the keys might start with "convFeat" or "convDec"
            #newKey = key.replace("convFeat.","").replace("convDec.","")
            newKey = key

            #Adding the noise
            if noise_init != 0:

                noise = torch.randn(params[key].size())

                if cuda:
                    noise = noise.cuda()

                params[key] += noise_init*0.5*torch.pow(params[key],2).sum()*noise

            if cuda:
                params[key] = params[key].cuda()

            if newKey in self.state_dict().keys():
                if newKey.find("dense") == -1:
                    self.state_dict()[newKey].data += params[key].data -self.state_dict()[newKey].data
            #else:
            #    print("Cannot find parameters {}".format(newKey))

class ClustDetectNet(nn.Module):
    '''A clustering-detecting network

    This module is made of a clustering network (which is a CNN module) and several detecting networks (which are all ConvFeatExtractor modules)
    The clustering network (CN) is an gating network and the detecting net (DN) are experts. The CN reads the image and decides which DN to use
    to determine if the image is from the class 0 or the class 1. The capacity of the DNs should be low so the CN is forced to specialise each
    DN to solve the task.

    Once all DN have produced their final feature maps, their are summed along the channel axis to produce only one final feature map.

    '''

    def __init__(self,inSize,inChan,args):
        '''
        Args:
            inSize (int) : the size of the side of an input image (the image is assumed to be squarred)
            inChan (int): the number of input channel
            args (Namespace): the namespace containing all the arguments required for training and building the network
        '''
        super(ClustDetectNet,self).__init__()

        self.decisionClu = args.decision_clu
        self.nbCl = int(args.clust)
        if type(args.cuda) is str:
            self.runCuda = True if args.cuda=="True" else False
        else:
            self.runCuda = args.cuda

        #------------------Building the encoder net--------------------------------#

        if int(args.encnblay) != 0:
            self.encoder = CAE(inSize=inSize,inChan=inChan,chan=args.encchan,hidd_repr_size=int(args.enchidd_repr_size),\
                               nbLay=int(args.encnblay),ker=int(args.encker),maxPl1=int(args.encmaxpl1),\
                               maxPl2=int(args.encmaxpl2),applyDropout2D=args.encdrop)

            encodedSize = int(args.enchidd_repr_size)
            encodedChan = 0
        else:
            self.encoder = None
            encodedSize = inSize
            encodedChan = inChan

        #------------------Building the clustering net-----------------------------#

        #If a dictionnary is used when building the net, the value will all be string
        #if it is an argparser object, the values will directly have the required type
        self.clDrop = str2bool(args.cldrop)  if (type(args.cldrop) is str) else args.cldrop
        self.cluDis = None

        self.clustNet = CNN(inSize=encodedSize,inChan=encodedChan,chan=int(args.clchan),\
                            avPool=False,nbLay=int(args.clnblayers),ker=int(args.clker),maxPl1=int(args.clmaxpoolsize),\
                            maxPl2=int(args.clmaxpoolsize_out),applyDropout2D=self.clDrop,nbOut=int(args.clust),\
                            applyLogSoftmax=False,nbDenseLay=int(args.clnb_denselayers),sizeDenseLay=int(args.clsize_denselayers))

        #------------------Building the detecting nets-----------------------------#

        self.avPool = str2bool(args.avpool) if (type(args.avpool) is str) else args.avpool
        self.deDrop = str2bool(args.dedrop)  if (type(args.dedrop) is str) else args.dedrop

        self.denb_denselayers = int(args.denb_denselayers)
        self.detectNets = nn.ModuleList([CNN(inSize=encodedSize,inChan=encodedChan,chan=int(args.dechan),\
                            avPool=args.avpool,nbLay=int(args.denblayers),ker=int(args.deker),maxPl1=int(args.demaxpoolsize),maxPl2=int(args.demaxpoolsize_out),\
                            applyDropout2D=self.deDrop,nbOut=2,nbDenseLay=int(args.denb_denselayers),sizeDenseLay=int(args.desize_denselayers)) for i in range(self.nbCl)])

        #------------------Building the final classifier---------------------------#

        if self.denb_denselayers == 0:
            if not self.avPool:
                linearInputSize = int(args.dechan)*((((inSize-args.deker+1)//int(args.demaxpoolsize))-args.deker+1)//int(args.demaxpoolsize_out))**2
                self.dense = nn.Linear(linearInputSize,2)
            else:
                self.dense = nn.Linear(int(args.dechan),2)
        else:
            self.dense = None

    def getClustWeights(self):
        '''Return the parameters of the clustering net'''
        params = []

        if self.encoder:
            for p in self.encoder.parameters():
                params.append(p)

        for p in self.clustNet.parameters():
            params.append(p)

        return (p for p in params)

    def getDetectWeights(self):
        '''Return the parameters of all the detecting network'''
        params = []

        if self.encoder:
            for p in self.encoder.parameters():
                params.append(p)

        for detNet in self.detectNets:
            for p in detNet.parameters():
                params.append(p)

        return (p for p in params)

    def setDetectWeights(self,params,cuda,noise_init):
        '''Set the parameters of the detecting nets and the final classifier using a dictionnary
        Args:
            see ConvFeatExtractor.setWeights() method
        '''

        #Set the parameters of the detecting nets
        for detNet in self.detectNets:
            detNet.setWeights(params,cuda,noise_init)

        #Set the parameters of the final classifier
        for key in params.keys():
            if key in self.state_dict().keys():
                self.state_dict()[key].data += params[key].data - self.state_dict()[key].data

    def forward_detect(self,x,cluDis=None):
        ''' Compute the output of the clustering network and of the detecting networks
        Args:
            x (torch.autograd.variable.Variable): the batch of images to process.
            cluDis (torch.autograd.variable.Variable): the batch of discrete distribution (i.e. list of vectors) to use to weight the output
                of the detecting network. If set to None, this will be computed by the clustering network
        Returns:
            x (torch.autograd.variable.Variable): the processed batch
            netshape (string): a string indicating the shape of activations and name of operation at each layer
            actArr (list) the list of activations array of each layer
         '''

        #This string will contain the shape of every intermediary tensor used during computing
        netshape ="in : "+str(x.size())+"\n"

        if cluDis is None:
            #Computing the clustNet output
            cluDis,actArr = self.clustNet(x)
            netshape +="clustNet: "+str(cluDis.size())+"\n"
            cluDis = F.softmax(cluDis, dim=1)
        else:
            actArr = []
        self.cluDis = cluDis

        if self.runCuda:
            self.cluDis = self.cluDis.cuda()

        #The decision is stochastic during train and eval
        if self.decisionClu == 'FS':
            cluDis = oneHotActivation.apply(cluDis)

        #The decision is stochastic during train and deterministic during eval
        elif self.decisionClu == 'DE':
            if self.training:
                cluDis = oneHotActivation.apply(cluDis)
            else:
                cluDis = oneHotActivationDeterministic.apply(cluDis)

        #The decision is always deterministic
        elif self.decisionClu == 'FD':
            cluDis = oneHotActivationDeterministic.apply(cluDis)

        #The decision is always soft
        elif self.decisionClu != 'SD':

            raise ValueError("Unknown decision mode : decisionClu = {}. Must be one of \'FS\' (full stochastic), \'DE\' (deterministic eval),\
            \'FD\' (full deterministic), \'SD\' (soft decision) )".format(self.decisionClu))

        #Computing the output the each detectNet
        #The input of each net is multiplied by the corresponding coefficient generated by the clustering net
        #If a hard decision is made (i.e. a one-hot vector is produced by the clustering net), all the inputs
        #of the detecting nets will be zero tensor except the one chosen by the one-hot vector, which receives
        #the normal image.
        outputs = []
        acts = []
        for i in range(len(self.detectNets)):

            if len(x.size()) == 4:
                coeff = cluDis[:,i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            else:
                coeff = cluDis[:,i].unsqueeze(1)

            coeff = coeff.expand(x.size())

            output,act = self.detectNets[i](x*coeff)
            outputs.append(output.unsqueeze(0))
            acts.append(act)

        #Concatenating the outputs of each detectNets
        actArr.append(acts)
        x = torch.cat(outputs, dim=0)
        netshape +="detectNet: "+str(x.size())+"\n"
        return x,netshape,actArr

    def forward_dense(self,x,netshape,actArr):
        ''' Compute the output of the dense layer and apply the log-softmax
        Args:
            x (torch.autograd.variable.Variable): the batch of images to process.
            netshape (string): a string which will be appended with the shape of activations and name of operation at each layer
            actArr (list) a list to which will be appended the activations array of each layer
        Returns:
            x (torch.autograd.variable.Variable): the processed batch
            netshape (string): a string indicating the shape of activations and name of operation at each layer. This string starts
                with the content in the string passed as parameter
            actArr (list) the list of activations array of each layer. This list starts with the elements of the list passed as parameters

         '''

        #Flattening the tensor
        if self.denb_denselayers == 0:
            x = x.view(x.size()[0],x.size()[1]*x.size()[2]*x.size()[3])

            #Softmax layer
            x = self.dense(x)

        netshape +="dense : "+ str(x.size())+"\n"
        actArr.append(x)
        x = F.log_softmax(x, dim=1)

        return x,netshape,actArr

    def forward(self,x,cluDis=None):
        '''Compute the output the clustDetectNet given a batch of images
        Args:
            x (torch.autograd.variable.Variable): the batch of images to process. The dimensions should be (batch_size,channel_number,width,height)
            cluDis (torch.autograd.variable.Variable): the batch of discrete distribution (i.e. list of vectors) to use to weight the output
                of the detecting network. If set to None, this will be computed by the clustering network
        Returns:
            x (torch.autograd.variable.Variable): the batch of predictions
            actArr (list) the list of activations array of each layer.
        '''
        if self.encoder:
            x,_,_ = self.encoder.computeHiddRepr(x)
        x,netshape,actArr = self.forward_detect(x,cluDis=cluDis)
        if self.avPool:
            x = x.sum(dim=-1,keepdim=True).sum(dim=-2,keepdim=True)
        x = torch.sum(x,dim=0)
        netshape +="feature map sum: "+str(x.size())+"\n"
        x,netshape,actArr = self.forward_dense(x,netshape,actArr)
        return x,actArr

class FullClustDetectNet(ClustDetectNet):
    '''A clustering-detecting network

    A child class of ClustDetectNet. It should contains one detecting network (DN) for every class in the dataset.
    The DNs are separated in two groups : half of them detect positive classes (the positive DNs) and the rest detect negative classes
    (the negative DNs).

    Once all DN have produced their final feature maps, the feature maps from negative DNs are multiplied by -1 and then
    all the feature maps are summed along the channel axis to produce only one final feature map.

    '''

    def __init__(self,inSize,inChan,args):
        '''
        Args:
            inSize (int) : the size of the side of an input image (the image is assumed to be squarred)
            inChan (int): the number of input channel
            args (Namespace): the namespace containing all the arguments required for training and building the network
        '''
        super(FullClustDetectNet,self).__init__(inSize,inChan,args)

        if args.clust%2 != 0: raise ValueError("The number of cluster should be an even number. Got {}".format(args.clust))

        if self.avPool != True:

            linearInputSize = int(args.dechan)*((inSize//int(args.demaxpoolsize))//int(args.demaxpoolsize_out))**2

            #The dense layer input is twice the size of a regular ClustDetectNet because the FullClustDetectNet
            #has two feature map before this layer : the positive maps and the negative maps. A regular ClustDetectNet
            #has only one feature map at this point.
            self.dense = nn.Linear(linearInputSize*2,2)
        else:
            self.dense = nn.Linear(int(args.dechan)*2,2)

    def setDetectWeights(self,params,cuda,noiseAmount,positive):
        '''Set the parameters of the detecting nets and the final classifier using a dictionnary
        Args:
            see ConvFeatExtractor.setWeights() method
        '''

        for key in params.keys():

            if key.find("dense") == -1:


                #positive is a boolean indicating if the parameters are to initialize the positive detecting net or not
                nbChannel = self.state_dict()["detectNets.0."+key.replace("convFeat.","")].size(0)

                if positive:
                    #Set the parameters of the positive detecting nets
                    for detNet in self.detectNets[:nbChannel//2]:
                        detNet.setWeights(params,cuda,noiseAmount)
                else:
                    #Set the parameters of the negative detecting nets
                    for detNet in self.detectNets[nbChannel//2:]:
                        detNet.setWeights(params,cuda,noiseAmount)

    def forward(self,x):
        '''Compute the output the full clustering net given a batch of images
        Args:
            x (torch.autograd.variable.Variable): the batch of images to process. The dimensions should be (batch_size,channel_number,width,height)
        Returns:
            x (torch.autograd.variable.Variable): the batch of predictions
            actArr (list) the list of activations array of each layer.
        '''
        x,_ = self.encoder(x)

        x,netshape,actArr = self.forward_detect(x)

        #Multiplying feature maps from negative DNs by -1
        coeff = torch.ones((self.nbCl))
        coeff[int(self.nbCl/2):] = -1
        coeff = coeff.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        coeff = coeff.expand(-1,x.size(1),x.size(2),x.size(3),x.size(4))
        x = x * coeff

        #Summing along the channel axis
        posFeatMaps = torch.sum(x[:self.nbCl//2],dim=0)
        negFeatMaps = torch.sum(x[self.nbCl//2:],dim=0)

        x = torch.cat([posFeatMaps,negFeatMaps],dim=1)

        #x = torch.sum(x,dim=0)

        netshape +="feature map sum: "+str(x.size())+"\n"

        x,netshape,actArr = self.forward_dense(x,netshape,actArr)

        return x,actArr

def netMaker(args):
    '''Build a network
    Args:
        args (Namespace): the namespace containing all the arguments required for training and building the network
    Returns:
        the built network
    '''

    #Setting the size and the number of channel depending on the dataset
    if args.dataset == "MNIST":
        inSize = 28
        inChan = 1
    elif args.dataset == "CIFAR10":
        inSize = 32
        inChan = 3
    else:
        raise("netMaker: Unknown Dataset")

    #During training a clustering-detecting network is built
    if not args.pretrain:

        if not args.pretrain_cae:
            if args.full_clust:
                net = FullClustDetectNet(inSize=inSize,inChan=inChan,args=args)
            else:
                net = ClustDetectNet(inSize=inSize,inChan=inChan,args=args)
        else:
            net = CAE(inSize=inSize,inChan=inChan,chan=args.encchan,hidd_repr_size=args.enchidd_repr_size,nbLay=args.encnblay,\
                      ker=args.encker,maxPl1=args.encmaxpl1,maxPl2=args.encmaxpl2,applyDropout2D=args.encdrop)
    #During pretraining a cnn is built
    else:
        net = CNN(inSize=inSize,inChan=inChan,chan=args.dechan,avPool=args.avpool,nbLay=args.denblayers,\
                  ker=args.deker,maxPl1=args.demaxpoolsize,maxPl2=args.demaxpoolsize_out,applyDropout2D=args.dedrop,nbOut=2,\
                  applyLogSoftmax=True,nbDenseLay=args.denb_denselayers,sizeDenseLay=args.desize_denselayers)

    return net

if __name__ == '__main__':

    convFeat = ConvFeatExtractor(3,1,5,3,2,2,True,outChan=2)

    x =  convFeat(torch.rand((10,3,16,16)))
    print(x[0])
    for act in x[1]:
        print(act.size())
    print(x[2]["inds1"])
