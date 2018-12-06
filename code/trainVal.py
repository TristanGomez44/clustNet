import sys
from args import ArgReader
from args import str2bool
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from torch.autograd import Variable
import netBuilder
import os
import dataLoader
import configparser
import torch.nn.functional as F
import vis
from torch.distributions import Bernoulli
class GradNoise():
    '''A class to add gaussian noise in weight update

    To be used with a pytorch hook so this function is called every time there is a weight update

    '''

    def __init__(self,ampl=0.1):
        '''
        Args:
            ampl (float): the ratio of the noise norm to the gradient norm
        '''

        self.ampl=ampl

    def __call__(self,grad):
        '''
        Args:
            grad (torch.autograd.variable.Variable): the gradient of the udpate
        Returns:
            The gradient with added noise
        '''

        self.noise = np.random.normal(size=grad.detach().cpu().numpy().shape)
        gradNorm = torch.sqrt(torch.pow(grad,2).sum()).item()
        noise =self.ampl*gradNorm*self.noise

        if grad.is_cuda:
            return grad + torch.tensor(noise).cuda().type("torch.cuda.FloatTensor")
        else:
            return grad + torch.tensor(noise)

def trainCAE(cae,optimizerCAE,train_loader, epoch, args):
    '''Train a convolutional autoencoder network

    After having run the net on every image of the train set,
    its state is saved in the nets/NameOfTheExperience/ folder

    Args:
        cae (CAE): a CAE module (as defined in netBuilder)
        optimizerCAE (torch.optim): the optimizer to train the network
        train_loader (torch.utils.data.DataLoader): the loader to generate batches of train images
        epoch (int): the current epoch number
        args (Namespace): the namespace containing all the arguments required for training and building the network
    '''

    cae.train()
    for batch_idx, (data, _) in enumerate(train_loader):

        if args.cuda:
            data= data.cuda()
        data = Variable(data)

        optimizerCAE.zero_grad()

        flattData = data[:,0].view(data.size(0),-1)

        output = cae(data)


        loss = F.mse_loss(output, data)

        loss.backward()

        optimizerCAE.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

        torch.save(cae.state_dict(), "../nets/{}/cae{}_epoch{}".format(args.exp_id,args.ind_id, epoch))

def testCAE(cae,test_loader,epoch, args,imgNbToWrite=10):
    '''Test a convolutional autoencoder network
    Compute the accuracy and the loss on the test set and write every output score of the net in a csv file

    Args:
        cae (CAE): a CAE module (as defined in netBuilder)
        test_loader (torch.utils.data.DataLoader): the loader to generate batches of test images
        epoch (int): the current epoch number
        args (Namespace): the namespace containing all the arguments required for training and building the network
    '''

    cae.eval()

    test_loss = 0

    #The header of the csv file is written after the first test batch
    firstTestBatch = True

    for data, _ in test_loader:

        if args.cuda:
            data = data.cuda()
        data = Variable(data)

        output = cae(data)

        test_loss +=  F.mse_loss(output, data, size_average=False).data.item()

    #Using the last batch to write reconstructed images
    for i in range(imgNbToWrite):
        vis.writeImg("../vis/{}/cae{}_{}.jpg".format(args.exp_id,args.ind_id,i),data[i,0].detach().cpu().numpy())
        vis.writeImg("../vis/{}/cae{}_{}_epoch{}_Rec.jpg".format(args.exp_id,args.ind_id,i,epoch),output[i,0].detach().cpu().numpy())

    #Error per image
    test_loss /= len(test_loader.dataset)
    #Error per pixel
    test_loss /= data.size(-1)*data.size(-2)

    #Error relative to pixel amplitude
    #2.8215 (-0.4242) is the maximum (minimum) value the pixels can take,
    #given the normalization done in the data loader
    test_loss /= 2.8215-(-0.4242)
    test_loss *= 100

    print('\nTest set: Average Error per pixel: {:.4f}%'.format(test_loss))

def trainDetect(detectNet,optimizerDe,train_loader, epoch, args,classToFind):
    '''Train a detecting network

    After having run the net on every image of the train set,
    its state is saved in the nets/NameOfTheExperience/ folder

    Args:
        detectNet (CNN): a CNN module (as defined in netBuilder) with two outputs
        optimizerDe (torch.optim): the optimizer to train the network
        train_loader (torch.utils.data.DataLoader): the loader to generate batches of train images
        epoch (int): the current epoch number
        args (Namespace): the namespace containing all the arguments required for training and building the network
        classToFind (list): the list of class index to detect
    '''

    detectNet.train()
    for batch_idx, (data, origTarget) in enumerate(train_loader):

        #The labels provided by the data loader are merged to only 2 classes. The first class is defined by the class
        #index in classToFind and the second class is defined by the left labels
        target = merge(origTarget,args.reverse_target)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizerDe.zero_grad()

        output,_ = detectNet(data)

        loss = F.nll_loss(output, target)
        loss.backward()

        optimizerDe.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

        torch.save(detectNet.state_dict(), "../nets/{}/detectNet{}_epoch{}".format(args.exp_id,args.ind_id, epoch))

def testDetect(detectNet,test_loader,epoch, args,classToFind):
    '''Test a detecting network
    Compute the accuracy and the loss on the test set and write every output score of the net in a csv file

    Args:
        detectNet (CNN): a CNN module (as defined in netBuilder) with two outputs
        test_loader (torch.utils.data.DataLoader): the loader to generate batches of test images
        epoch (int): the current epoch number
        args (Namespace): the namespace containing all the arguments required for training and building the network
        classToFind (list): the list of class index to detect
    '''

    detectNet.eval()

    test_loss = 0
    correct = 0

    #The header of the csv file is written after the first test batch
    firstTestBatch = True

    for data, origTarget in test_loader:

        #The labels provided by the data loader are merged to only 2 classes. The first class is defined by the class
        #index in classToFind and the second class is defined by the left labels
        target = merge(origTarget,args.reverse_target)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output,_ = detectNet(data)

        test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        #Writing the CSV files in the results/NameOfTheExperience/ folder
        writeCSV(args,epoch,firstTestBatch,list(origTarget.cpu().numpy()), list(target.cpu().numpy()),list(output),phase="test")
        firstTestBatch = False

    #Print the results
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def train(clustDetectNet,optimizerCl, optimizerDe,train_loader, epoch, args,classToFind):
    '''Train a clustering-detecting network

    After having run the net on every image of the train set,
    its state is saved in the nets/NameOfTheExperience/ folder

    Write every output score of the net in a csv file. The distrutions outputed by the clustering network are
    also written in the same file.

    Can add several term to the loss function :

    - entweig: reward high entropy of the outputs
    - filter_dis : reward large cosine distance between the filters
    - clustdivers: reward diversity between the distributions proposed by the clustering network
    - featmap_entr: reward low entropy of the last layer feature map for each detecting net and each class (positive and negative) separately
    - featmap_var: reward low variance of the last layer feature map for each detecting net and each class (positive and negative) separately

    Args:
        clustDetectNet (ClustDetectNet): a ClustDetectNet module (as defined in netBuilder) with two outputs
        optimizerCl (torch.optim): the optimizer to train the clustering network
        optimizerDe (torch.optim): the optimizer to train the detecting networks
        train_loader (torch.utils.data.DataLoader): the loader to generate batches of train images
        epoch (int): the current epoch number
        args (Namespace): the namespace containing all the arguments required for training and building the network
        classToFind (list): the list of class index to detect
    '''

    clustDetectNet.train()

    all_targ = []
    all_clust = []
    firstTrainBatch = True
    correct = 0

    nb_batch = len(train_loader.dataset)//args.batch_size

    for batch_idx, (data, origTarget) in enumerate(train_loader):

        target = merge(origTarget)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizerCl.zero_grad()
        optimizerDe.zero_grad()

        if args.clu_train_mode == "joint":
            all_targ_tmp,all_clust_tmp = jointTrain(clustDetectNet,args,train_loader,epoch,batch_idx,data,origTarget,target,optimizerDe,optimizerCl,correct,firstTrainBatch)
        elif args.clu_train_mode == "separated":
            all_targ_tmp,all_clust_tmp = separatedTrain(clustDetectNet,args,train_loader,epoch,batch_idx,data,origTarget,target,optimizerDe,optimizerCl,correct,firstTrainBatch)
        else:
            raise ValueError("Unknown training mode : {}. Available training modes are \'joint\' and \'separated\'".format(args.clu_train_mode))

        all_targ.extend(all_targ_tmp)
        all_clust.extend(all_clust_tmp)

    torch.save(clustDetectNet.state_dict(), "../nets/{}/clustDetectNet{}_epoch{}".format(args.exp_id,args.ind_id, epoch))

    #Computing confusion matrix
    conf_mat = computeConfMat(all_clust,all_targ,args.exp_id,classToFind,args.clust,"clustDetectNet"+str(args.ind_id)+"_epoch"+str(epoch)+"_train.png")


def jointTrain(clustDetectNet,args,train_loader,epoch,batch_idx,data,origTarget,target,optimizerDe,optimizerCl,correct,firstTrainBatch):

    #The forward method of the clustDetectNet class will store the output
    #of the clusterNet after each forward pass
    output,actArr = clustDetectNet(data)

    #Getting the clusterNet output
    cluDis = clustDetectNet.cluDis
    #Computing the argmax of each sample
    _,argmax_x_clDis = torch.max(cluDis,dim=1)

    #Collecting the labels
    all_targ = list(origTarget.cpu().numpy())
    all_clust = list(argmax_x_clDis.cpu().numpy())

    loss = F.nll_loss(output, target)

    #Add some terms to the loss function to change the behavior of the net during training
    loss = addLossTerms(loss,args.clust,args.denblayers,args.harddecision,args.entweig,args.filter_dis,args.clustdivers,args.featmap_entr,args.featmap_var)

    loss.backward()
    #print(loss.grad)

    if args.batch_period == 0:
        optimizerDe.step()
        optimizerCl.step()
    else:
        if batch_idx % (2*args.batch_period) > args.batch_period-1:
            optimizerDe.step()
        else:
            optimizerCl.step()

    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    writeCSV(args,epoch,firstTrainBatch,list(origTarget.cpu().numpy()), list(target.cpu().numpy()),list(output), list(cluDis),phase="train")
    firstTrainBatch = False

    if batch_idx % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data.item()))

    return all_targ,all_clust
def computeRandProb(rand_prop_val,rand_prop_epo, epoch):

    if len(rand_prop_epo) != len(rand_prop_val):
        raise ValueError("rand_prop_epo length is different from rand_prop_epo length")

    if epoch< rand_prop_epo[0]:
        prop = rand_prop_val[0]
    else:
        i=0
        intervalFound =False
        while i<len(rand_prop_epo)-1 and not intervalFound:
            if rand_prop_epo[i] <= epoch and epoch < rand_prop_epo[i+1]:
                intervalFound = True
            else:
                i+=1

        if intervalFound:
            prop = rand_prop_val[i]+(rand_prop_val[i+1]-rand_prop_val[i])*(epoch-rand_prop_epo[i])/(rand_prop_epo[i+1]-rand_prop_epo[i])
        else:
            prop = rand_prop_val[-1]

    return prop

def separatedTrain(clustDetectNet,args,train_loader,epoch,batch_idx,data,origTarget,target,optimizerDe,optimizerCl,correct,firstTrainBatch):

    """Train separately the clustnet and the detectnets

    The training of the clustering network goes like this :
    For each image, we need to find a detectnet whose index will be used as target for the clustnet
    - With probability p, the detectnet chosen will be the one giving the smallest loss
    - With probability 1-p, it will be a random decision : the chosen detectnet is sampled from uniform distribution

    """

    if batch_idx % (2*args.batch_period) > args.batch_period-1:

        #print("Training the clustnet")

        #Determining the images for which a random detectnet will be asigned
        randProb = torch.tensor(computeRandProb(args.rand_prop_val_sched,args.rand_prop_epo_sched,epoch))
        randomDecision = torch.bernoulli(randProb.expand(len(data))).type(torch.LongTensor)

        if batch_idx==2:
            print(randProb)

        #Choosing the net that the clustnet has to choose for each image
        #For each image, with propability p, the net will be the one giving the smallest loss
        #And with probability 1-p, the net will be random
        clustTargets = torch.zeros_like(target,dtype=torch.long)

        ##################Assigning the images with the smallest loss criteria####################

        dataSmallestLoss = masked_index(data, dim=0, mask=randomDecision)

        if int(randomDecision.sum()) != 0:

            imageShape = dataSmallestLoss[0].size()

            #Add another axis and repeat eacj image on this axis
            dataExp = dataSmallestLoss.unsqueeze(1)
            dataExp = dataExp.repeat(1,args.clust,1,1,1)
            #Flattening along that axis
            dataExp = dataExp.view(args.clust*int(randomDecision.sum()),imageShape[0],imageShape[1],imageShape[2])

            cluDisExp = torch.eye(args.clust,dtype=torch.float).repeat(int(randomDecision.sum()),1)

            if args.cuda:
                cluDisExp = cluDisExp.cuda()

            targetExp = masked_index(target, dim=0, mask=randomDecision)
            targetExp = targetExp.unsqueeze(1).repeat(1,args.clust).view(args.clust*int(randomDecision.sum()))

            output,_ = clustDetectNet(dataExp,cluDis=cluDisExp)
            loss = F.nll_loss(output, targetExp,reduce=False).view(int(randomDecision.sum()),args.clust)

            clustTargetSmallestLoss = torch.min(loss,dim=1)[1]
            clustTargets[torch.nonzero(randomDecision)[:,0]] = clustTargetSmallestLoss

        ##################Assigning the other images with a random criteria########################

        if int((1-randomDecision).sum()) > 0:

            clustTargetRandom = torch.Tensor(int((1-randomDecision).sum())).random_(0,args.clust-1).long()
            if args.cuda:
                clustTargetRandom = clustTargetRandom.cuda()
            clustTargets[torch.nonzero(1-randomDecision)[:,0]] = clustTargetRandom

        if args.cuda:
            clustTargets = clustTargets.cuda()

        #Process the data
        if clustDetectNet.encoder:
            data,_,_ = clustDetectNet.encoder.computeHiddRepr(data)

        cluDis,actArr = clustDetectNet.clustNet(data)
        cluDis = F.log_softmax(cluDis, dim=1)

        _,argmax_x_clDis = torch.max(cluDis,dim=1)

        loss = F.nll_loss(cluDis, clustTargets)
        loss.backward()
        optimizerCl.step()

        all_targ = []
        all_clust = []

    else:
        #print("Training the detectnets")


        output,actArr = clustDetectNet(data)

        #Getting the clusterNet output
        cluDis = clustDetectNet.cluDis
        #Computing the argmax of each sample
        _,argmax_x_clDis = torch.max(cluDis,dim=1)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizerDe.step()

        #Collecting the labels
        all_targ = list(origTarget.cpu().numpy())
        all_clust = list(argmax_x_clDis)

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        writeCSV(args,epoch,firstTrainBatch,list(origTarget.cpu().numpy()), list(target.cpu().numpy()),list(output), list(cluDis),phase="train")

    if batch_idx % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data.item()))

    return all_targ,all_clust
def addLossTerms(loss,clust,denblayers,harddecision,entweig,filter_dis,clustdivers,featmap_entr,featmap_var):

    if not harddecision and entweig != 0 :
        loss += entweig * (-cluDis*torch.log(cluDis)).mean()

    if filter_dis != 0:

        #Choosing one detect net
        #Its last layer will be moved away from the others net's last layer
        ind = np.random.randint(low=0,high=clust-1)

        #Getting last layer filters
        filtersW = list(getattr(clustDetectNet,"deConvs").parameters())[2*denblayers-2]
        filtersB = list(getattr(clustDetectNet,"deConvs").parameters())[2*denblayers-1]

        #Separating them by channel
        filtersW = filtersW.view(clust,-1,filtersW.size(1),filtersW.size(2),filtersW.size(3))
        filtersB = filtersB.view(clust,-1)

        #The filters are permutated randomly
        #By doing so, each filter will be progressively made distant
        #from each other net filters
        choosedWReap = filtersW[ind][0].repeat(filtersW.size(0),1,1,1)
        choosedBReap = filtersB[ind].repeat(filtersB.size(0))

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        totalDist = cos(choosedWReap.view(-1),filtersW.view(-1))
        totalDist += cos(choosedBReap.view(-1),filtersB.view(-1))

        loss -=filter_dis*totalDist

    if clustdivers != 0:

        #Computing frequencies of clusters
        frequ = cluDis.sum(dim=0)/cluDis.size(0)
        loss -= clustdivers*(frequ*torch.log(frequ+0.0000001)).mean()

    if featmap_entr != 0 or featmap_var != 0:

        for i in range(clust):

            #print(torch.tensor(target))
            featMaps = list(filter(lambda x:x.sum() != 0,actArr[-3][:,i]))

            pos = masked_index(actArr[-3][:,i],0,torch.tensor(target).long())
            neg = masked_index(actArr[-3][:,i],0,1-torch.tensor(target).long())

            pos = masked_index(pos,0,(pos.sum(dim=1).sum(dim=1) != 0).long())
            neg = masked_index(neg,0,(neg.sum(dim=1).sum(dim=1) != 0).long())

            if featmap_entr != 0:
                if pos.nelement() != 0:
                    loss += -featmap_entr*(pos*torch.log(pos+0.0000001)).mean()
                if neg.nelement() != 0:
                    #print(neg.size())
                    loss += -featmap_entr*(neg*torch.log(neg+0.0000001)).mean()
            else:
                if pos.nelement() != 0:
                    loss += featmap_var*(pos.std(dim=0)).mean()
                if neg.nelement() != 0:
                    #print(neg.size())
                    loss += featmap_var*(neg.std(dim=0)).mean()

    return loss

def test(clustDetectNet,test_loader,epoch, args,classToFind):
    '''Test a clustering-detecting network
    Compute the accuracy and the loss on the test set and write every output score and clustering score of the net in a csv file

    Args:
        clustDetectNet (ClustDetectNet): a ClustDetectNet module (as defined in netBuilder) with two outputs
        test_loader (torch.utils.data.DataLoader): the loader to generate batches of test images
        epoch (int): the current epoch number
        args (Namespace): the namespace containing all the arguments required for training and building the network
        classToFind (list): the list of class index to detect
    '''

    clustDetectNet.eval()

    test_loss = 0
    correct = 0
    #During the first test batch, the csv file has to be created
    firstTestBatch = True
    all_targ = []
    all_clust = []

    for data, origTarget in test_loader:

        target = merge(origTarget)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output,_ = clustDetectNet(data)

        #Extracting the clustering score i.e. the vector indicating which detecting
        #nets to use to process the image
        cluDis = clustDetectNet.cluDis

        #Computing the argmax of each sample
        _,argmax_x_clDis = torch.max(cluDis,dim=1)

        all_targ += list(origTarget.cpu().numpy())
        all_clust += list(argmax_x_clDis)

        test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        #Writing the CSV files
        writeCSV(args,epoch,firstTestBatch,list(origTarget.cpu().numpy()), list(target.cpu().numpy()),list(output), list(cluDis),phase="test")
        firstTestBatch = False

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    #Computing confusion matrix
    conf_mat = computeConfMat(all_clust,all_targ,args.exp_id,classToFind,args.clust,"clustDetectNet"+str(args.ind_id)+"_epoch"+str(epoch)+"_test.png")

def masked_index(input, dim, mask):
    '''Select elements of a pytorch array using a boolean mask
    Args:
        input (torch.autograd.variable.Variable): the pytorch array
        dim (int): the dimension at which to Select
        mask (list): a boolean list indicating which element has to be selected
    Return the array with only the selected elements
    '''
    assert len(mask.size())==1 and input.size(dim)==mask.size(0),'{}!=1 or {}!={}'.format(len(mask.size()), input.size(dim), mask.size(0))
    indices = torch.arange(0,mask.size(0))[mask.data.type("torch.ByteTensor")].long()
    return input[indices]

def merge(target,reverse_target=False,listLabels = [0,1,2,3,4]):

    '''Merge classes in a batch of targets. This f

    This function defines the positive and the negative class

    All the labels in the batch will be replaced by 1 if the label is in listLabels
    and 0 else.

    Args:
        target (torch.autograd.variable.Variable): the vector of targets
        listLabels (list): the list of labels to replace by the positive label (which is 1).

    Returns:
        The new target batch with only two classes
    '''

    newTarg = []
    for i in range(len(target)):
        newTarg.append(target[i] in listLabels)

    if reverse_target:
        newTarg = 1-np.array(newTarg)

    return torch.LongTensor(newTarg)

def computeConfMat(all_clust,all_targ, exp_id,classToFind,nbClusts,heatmapFileName=None):

    '''Compute a rectangular confusion matrix.

    This matrix indicates proportion of each target class assigned to each cluster

    The rows of the matrix indicate the clusters indexs and the columns indicate the classes in the dataset
    There is as many rows (clusters) as classes to find
    There is as many columns as classes in the dataset

    Args:
        all_clust (list): list of clusters indexs choosed by the clustering network
        all_targ (list): list of target index (must be as long as all_clust)
        heatmapFileName (str): the name of the heatmap file. If none, the file won't be created
        exp_id (str): the name of the experiment
        classToFind (list): the list of class index defining the positive class.
        nbClusts (int): the total number of clusters
    Returns: a rectangular confusion matrix indicating the proportion of each target class assigned to each clusters
    '''

    #print(all_targ)
    nbClasses = len(list(set(list(all_targ))))

    #Indicates which class is asigned to each column of the matrix
    #The first classes in the dict are the classes to detect
    #the classes just after are the classes not to detect
    classDic = {}
    reversClassDic = {}
    #Adding the class to find
    for i in range(len(classToFind)):
        classDic.update({classToFind[i]: i})
        reversClassDic.update({i:classToFind[i]})
    #Adding the classes not to find
    classNotToFind = list(set(list(all_targ)) - set(classToFind))
    for i in range(len(classToFind),len(classToFind)+len(list(classNotToFind))):
        classDic.update({classNotToFind[i-len(classToFind)]: i})
        reversClassDic.update({i:classNotToFind[i-len(classToFind)]})

    #Computing the matrix
    mat = np.zeros((nbClusts,nbClasses))

    for i in range(len(all_targ)):
        try:
            mat[all_clust[i],classDic[all_targ[i]]] += 1
        except IndexError:
            print("Index error")
            print(mat.shape)
            print(all_clust[i],classDic[all_targ[i]])
            sys.exit(0)
    #If the filename is not set, the heatmap will not be written on the disk
    if heatmapFileName:
        mat = mat/mat.sum(axis=0)

        #Plotting the heat map
        heatmap = plt.figure()
        ax1 = heatmap.add_subplot(111)
        plt.xticks(np.arange(nbClasses),[reversClassDic[i] for i in range(nbClasses)])
        plt.imshow(mat, cmap='gray', interpolation='nearest')
        plt.savefig('../vis/'+str(exp_id)+"/"+heatmapFileName)

    return mat

def writeCSV(args,epoch,firstTestBatch, all_origTarg, all_targ, all_predScores,all_clustScores=None,phase="train"):
    '''Write a csv file with the targets (going from 0 to 9), the binary targets (0 or 1), and the scores predicting
    the binary target for the current batch. Can also writes the scores produced by the clustering network.

    Every time a batch is processed during an epoch, this function is called and the results of the batch processing
    printed in the csv file, just after the results from the last batch.

    Args:
        args (Namespace): the namespace containing all the arguments required for training and building the network
        epoch (int): the epoch number
        firstTestBatch (bool): Indicates if the current batch is the first of its epoch. At the first batch of an epoch, the csv file has to be created.
        all_origTarg (list): the list of targets (going from 0 to 9).
        all_targ (list): the list of binary targets (0 or 1).
        all_predScores (list): the list of scores predicting the binary targets.
        all_clustScores (list): the list of cluster scores. Can be None if the net doesn't produce clustering scores
        phase (str): indicates the phase the network is currently in (can be \'train\', \'validation\' or \'test\')
    '''

    if phase == "test":
        filePath = "../results/"+str(args.exp_id)+"/all_scores_net"+str(args.ind_id)+"_epoch"+str(epoch)+".csv"
    else:
        filePath = "../results/"+str(args.exp_id)+"/all_scores_net"+str(args.ind_id)+"_epoch"+str(epoch)+"_train.csv"

    #Create the csv file and write its header if this is the first batch of the epoch
    if firstTestBatch:
        with open(filePath, "w") as text_file:
            if all_clustScores:
                print("#origTarg,targ,pred[{}],clust[{}]".format(len(all_predScores[0]),len(all_clustScores[0])),file=text_file)
            else:
                print("#origTarg,targ,pred[{}]".format(len(all_predScores[0])),file=text_file)

    #Write all the targets, the binary targets, the clustering scores and the scores predicting the binary targets
    with open(filePath, "a") as text_file:
        for i in range(len(all_targ)):
            #Writing the target
            print(str(all_origTarg[i])+",",end="",file=text_file)
            print(str(all_targ[i])+",",end="",file=text_file)
            #Writing the prediction scores
            for j in range(len(all_predScores[i])):
                print(str(all_predScores[i][j].data.item())+",",end="",file=text_file)
            #Writing the cluster score
            if all_clustScores:
                for j in range(len(all_clustScores[i])):
                    print(str(all_clustScores[i][j].data.item())+",",end='',file=text_file)

            print("",file=text_file)

def findLastNumbers(weightFileName):
    '''Extract the epoch number of a weith file name.

    Extract the epoch number in a weight file which name will be like : "clustDetectNet2_epoch45".
    If this string if fed in this function, it will return the integer 45.

    Args:
        weightFileName (string): the weight file name
    Returns: the epoch number

    '''

    i=0
    res = ""
    allSeqFound = False
    while i<len(weightFileName) and not allSeqFound:
        if not weightFileName[len(weightFileName)-i-1].isdigit():
            allSeqFound = True
        else:
            res += weightFileName[len(weightFileName)-i-1]
        i+=1

    res = res[::-1]

    return int(res)

def get_OptimConstructor_And_Kwargs(optimStr,momentum):
    '''Return the apropriate constructor and keyword dictionnary for the choosen optimiser
    Args:
        optimStr (str): the name of the optimiser. Can be \'AMSGrad\', \'SGD\' or \'Adam\'.
        momentum (float): the momentum coefficient. Will be ignored if the choosen optimiser does require momentum
    Returns:
        the constructor of the choosen optimiser and the apropriate keyword dictionnary
    '''

    if optimStr != "AMSGrad":
        optimConst = getattr(optim,optimStr)
        if optimStr == "SGD":
            kwargs= {'momentum': momentum}
        elif optimStr == "Adam":
            kwargs = {}
        else:
            raise ValueError("Unknown optimisation algorithm : {}".format(args.optim))
    else:
        optimConst = optim.Adam
        kwargs = {'amsgrad':True}

    print("Optim is :",optimConst)

    return optimConst,kwargs

def initialize_Net_And_EpochNumber(net,pretrain,init,init_clu,init_enc,init_pos,init_neg,exp_id,ind_id,cuda,noise_init,netType):
    '''Initialize a clustering detecting network

    Can initialise with parameters from detecting network or from a clustering detecting network

    If init is None, the network will be left unmodified. Its initial parameters will be saved.

    Args:
        net (CNN or ClustDetectNet): the net to be initialised
        pretrain (boolean): if true, the net trained is a detectNet (can be used after to initialize the detectNets of a clustDetectNet)
        init (string): the path to the weigth for initializing the net with
        init_pos (string): the path to the weigth for initializing the positive detecting nets.
        init_neg (string): the path to the weigth for initializing the negative detecting nets.
        exp_id (string): the name of the experience
        ind_id (int): the id of the network
        cuda (bool): whether to use cuda or not
        noise_init (float): the proportion of noise to add when initialising the parameters
            of the detecting nets of a clustering detecting network. Ignored if the net to be trained
            is a detecting network
    Returns: the start epoch number
    '''

    #Initialize the clustering net with weights from a CAE encoder
    if not (init_clu is None):
        params = torch.load(init_clu)

        #Remove the tensor corresponding to decoder parameters
        keysToRemove = []
        for key in params.keys():
            if key.find("convDec") != -1 :
                keysToRemove.append(key)

        for key in keysToRemove:
            params.pop(key, None)

        net.clustNet.setWeights(params,cuda,noise_init)

    #Initialize the detect nets with weights from a supervised training
    if not (init is None):
        params = torch.load(init)

        #Setting parameters with parameters from a clustering detecting network
        if "clust" in os.path.basename(init) or (pretrain==True):
            net.load_state_dict(params)
            startEpoch = findLastNumbers(init)+1

        #Setting parameters with parameters from a detecting network
        else:
            startEpoch = 1
            net.setDetectWeights(params,cuda,noise_init)

    elif (not (init_pos is None)) or (not (init_neg is None)):

        #Initialize the positive detect nets with weights from a supervised training
        if not (init_pos is None):
            paramsPos = torch.load(init_pos)
            net.setDetectWeights(paramsPos,cuda, noise_init,positive=True)
            startEpoch = 1

        #Initialize the negative detect nets with weights from a supervised training
        if not (init_neg is None):
            paramsNeg = torch.load(init_neg)
            net.setDetectWeights(paramsNeg,cuda, noise_init,positive=False)
            startEpoch = 1

    #Starting a network from scratch
    else:
        #Saving initial parameters
        torch.save(net.state_dict(), "../nets/{}/{}{}_epoch0".format(exp_id,netType,ind_id))
        startEpoch = 1

    if init_enc:

        params = torch.load(init_enc)
        net.encoder.load_state_dict(params)

    return startEpoch


def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--noise', type=float, metavar='NOISE',
                        help='the amount of noise to add in the gradient of the clustNet (in percentage)(default: 0.1)')
    argreader.parser.add_argument('--entweig', type=float, default=0, metavar='ENTWEI',
                        help='the weight of the clusters entropy term in the cost function (default: 0)')
    argreader.parser.add_argument('--clustdivers', type=float, default=0, metavar='ENTWEI',
                        help='the weight of the clusters diversity term in the cost function (default: 0)')
    argreader.parser.add_argument('--filter_dis', type=float, default=0, metavar='FILDIS',
                        help='the weight of the filter distance term in the cost function (default: 0)')

    argreader.parser.add_argument('--featmap_entr', type=float, default=0, metavar='FEATENT',
                        help='the weight of the feature map entropy term in the cost function (default: 0)')
    argreader.parser.add_argument('--featmap_var', type=float, default=0, metavar='FEATVAR',
                        help='the weight of the feature map var term in the cost function (default: 0)')

    argreader.parser.add_argument('--optim', type=str, default="SGD", metavar='OPTIM',
                        help='the optimizer algorithm to use (default: \'SGD\')')
    argreader.parser.add_argument('--noise_init', type=float, default="0", metavar='NOISEINIT',
                        help='The percentage of noise to add (relative to the filter norm) when initializing detectNets with \
                        a pre-trained detectNet')

    argreader.parser.add_argument('--reverse_target',type=str2bool, default="False", help='To inverse the positive and the negative class. Useful to train a detectNet \
                        which will be later used to produce negative feature map')

    argreader.parser.add_argument('--clu_train_mode', type=str, default='joint', metavar='TRAINMODE',
                        help='Determines the cluster training mode. Can be \'joint\' or \'separated\' (default: \'joint\')')

    argreader.parser.add_argument('--rand_prop_val_sched', type=float, nargs='+',default=[0.9,0.5,0.1], metavar='RANDPROP_VAL_SCHED',help=')')
    argreader.parser.add_argument('--rand_prop_epo_sched', type=int, nargs='+',default=[0,1,2], metavar='RANDPROP_EPO_SCHED',help=')')

    argreader.parser.add_argument('--init', type=str,  default=None,metavar='N', help='the weights to use to initialize the detectNets')
    argreader.parser.add_argument('--init_clu', type=str,  default=None,metavar='N', help='the weights to use to initialize the clustNets')
    argreader.parser.add_argument('--init_enc', type=str,  default=None,metavar='N', help='the weights to use to initialize the encoder net')
    argreader.parser.add_argument('--init_pos',type=str, default=None,metavar='N', help='the weights to use to initialize the positive detectnets. Ignored when not training a full clust detect net')
    argreader.parser.add_argument('--init_neg', type=str,  default=None,metavar='N', help='the weights to use to initialize the negative detectNets. Ignored when not training a full clust detect net')

    argreader.parser.add_argument('--encapplyDropout2D', default=True,type=str2bool, metavar='N',help='whether or not to apply 2D dropout in the preprocessing net')


    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.clust < 2: raise ValueError("The number of cluster must be at least 2. Got {}".format(args.clust))
    train_loader,test_loader = dataLoader.loadData(args.dataset,args.batch_size,args.test_batch_size,args.cuda,args.num_workers)

    #The group of class to detect
    np.random.seed(args.seed)
    classes = [0,1,2,3,4,5,6,7,8,9]
    np.random.shuffle(classes)
    classToFind =  classes[0: args.clust]

    #The folders where the experience file will be written
    if not (os.path.exists("../vis/{}".format(args.exp_id))):
        os.makedirs("../vis/{}".format(args.exp_id))
    if not (os.path.exists("../results/{}".format(args.exp_id))):
        os.makedirs("../results/{}".format(args.exp_id))
    if not (os.path.exists("../nets/{}".format(args.exp_id))):
        os.makedirs("../nets/{}".format(args.exp_id))

    if args.pretrain:
        netType = "detectNet"
    elif args.pretrain_cae:
        netType = "cae"
    else:
        netType = "clustDetectNet"

    #Write the arguments in a config file so the experiment can be re-run
    argreader.writeConfigFile("../nets/{}/{}{}.ini".format(args.exp_id,netType,args.ind_id))

    #Building the net
    net = netBuilder.netMaker(args)

    if args.cuda:
        net.cuda()

    startEpoch = initialize_Net_And_EpochNumber(net,args.pretrain,args.init,args.init_clu,args.init_enc,args.init_pos,args.init_neg,\
                                                args.exp_id,args.ind_id,args.cuda,args.noise_init,netType)

    net.classToFind = classToFind

    #Getting the contructor and the kwargs for the choosen optimizer
    optimConst,kwargs = get_OptimConstructor_And_Kwargs(args.optim,args.momentum)

    #If no learning rate is schedule is indicated (i.e. there's only one learning rate),
    #the args.lr argument will be a float and not a float list.
    #Converting it to a list with one element makes the rest of processing easier
    if type(args.lr) is float:
        args.lr = [args.lr]

    if type(args.lr_cl) is float:
        args.lr_cl = [args.lr_cl]

    if (not args.pretrain) and (not args.pretrain_cae):

        #Adding a hook to add noise at every weight update
        if args.noise != 0:
            gradNoise = GradNoise(ampl=args.noise)
            for p in net.getClustWeights():
                p.register_hook(gradNoise)

        #Train and evaluate the clustering detecting network for several epochs
        lrCounter = 0



        for epoch in range(startEpoch, args.epochs + 1):

            #This condition determines when the learning rate should be updated (to follow the learning rate schedule)
            #The optimiser have to be rebuilt every time the learning rate is updated
            if (epoch-1) % ((args.epochs + 1)//len(args.lr)) == 0 or epoch==startEpoch:

                kwargs['lr'] = args.lr[lrCounter]
                print("Learning rate : ",kwargs['lr'])
                optimizerDe = optimConst(net.getDetectWeights(), **kwargs)

                kwargs['lr'] = args.lr_cl[lrCounter]
                print("Learning rate of clustNet: ",kwargs['lr'])
                optimizerCl = optimConst(net.getClustWeights(), **kwargs)

                if lrCounter<len(args.lr)-1:
                    lrCounter += 1

            train(net,optimizerCl, optimizerDe,train_loader,epoch, args,classToFind)
            test(net,test_loader,epoch, args,classToFind)

    else:
        print("Pretraining")

        if args.pretrain_cae:
            trainFunc = trainCAE
            testFunc = testCAE
            kwargsFunc = {}
        else:
            trainFunc = trainDetect
            testFunc = testDetect
            kwargsFunc = {"classToFind":classToFind}

        #Train and evaluate the detecting network for several epochs
        lrCounter = 0
        for epoch in range(startEpoch, args.epochs + 1):

            if (epoch-1) % ((args.epochs + 1)//len(args.lr)) == 0 or epoch==startEpoch:

                kwargs['lr'] = args.lr[lrCounter]
                print("Learning rate : ",kwargs['lr'])
                optimizerDe = optimConst(net.parameters(), **kwargs)

                if lrCounter<len(args.lr)-1:
                    lrCounter += 1

            trainFunc(net,optimizerDe,train_loader,epoch, args,**kwargsFunc)
            testFunc(net,test_loader,epoch, args,**kwargsFunc)

if __name__ == "__main__":
    main()
