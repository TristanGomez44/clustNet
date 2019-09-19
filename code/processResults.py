import sys
import os
import numpy as np
import glob
from numpy.random import shuffle
from numpy import genfromtxt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
import scipy.stats

import scipy as sp
import scipy.stats
import os
import math
import ast
import sys
import argparse
import configparser
from args import ArgReader

import netBuilder
import torch.nn.functional as F
import torch
import dataLoader
import torch.nn.functional as F
import vis
import trainVal
from sklearn.manifold import TSNE
import netBuilder

def weigthTrajectoryLength(args):

    weigthTrajectoryLength(args)

    #Count the number of net in the experiment
    netNumber = len(glob.glob("../nets/{}/*.ini".format(args.exp_id)))

    #Get and sort the experiment file
    weigFiles = sortExperiFiles("../nets/"+args.exp_id+"/clustDetectNet*_epoch*".format(args.exp_id),netNumber)
    paramDictPaths = sorted(glob.glob("../nets/"+str(args.exp_id)+"/*.ini"))

    #Getting the dataset and the boolean parameter inweig
    #Assuming the all the net in the exp have the same dataset
    #and the same value for the boolean parameter inweig
    config = configparser.ConfigParser()
    config.read(paramDictPaths[0])
    dataset = config['default']["dataset"]
    inweig = config['default']["inweig"]
    clust =  int(config['default']["clust"])

    for i in range(len(weigFiles)):

        net = netBuilder.netMaker(args)

        #Getting the parameters the net had just after initializing
        #The initial parameters of the net are used to know how strong is
        #their evolution
        net.load_state_dict(torch.load(weigFiles[i,0]))
        #for p in net.parameters():
        #    print(p.size())

        firstParams = np.array(list(net.parameters()))
        for j in range(len(firstParams)):
            firstParams[j] = np.array(firstParams[j].detach().numpy())

        #Initializing the old parameters, i.e. the parameters of the last epoch
        distArr = np.zeros((len(weigFiles[i]),len(firstParams)))

        oldParams = np.array(list(net.parameters()))
        for j in range(len(oldParams)):
            oldParams[j] = np.array(oldParams[j].detach().numpy())

        for j in range(1,len(weigFiles[i])):
            #Getting the parameters for the current epoch
            net.load_state_dict(torch.load(weigFiles[i,j]))

            newParams = np.array(list(net.parameters()))
            for k in range(len(newParams)):
                newParams[k] = np.array(newParams[k].detach().numpy())

            #Computing the difference between the last weights and the current weights
            diffArr = (newParams-oldParams)/firstParams

            #Computing the distance between the preceding weights and the current weights
            for k in range(len(diffArr)):
                distArr[j-1,k] = np.sqrt(np.power(diffArr[k],2).sum())

            #Updating the old parameters
            oldParams = np.array(list(net.parameters()))
            for k in range(len(oldParams)):
                oldParams[k] = np.array(oldParams[k].detach().numpy())

        # Building the mean roc curve of all the network in the experiment
        plot = plt.figure()
        ax1 = plot.add_subplot(111)
        plt.xlabel('Epoch');
        plt.ylabel('Distance')
        ax1.plot(distArr[:,0], 'yellow', label="Clust conv1")
        ax1.plot(distArr[:,2], 'orange', label="Clust conv2")
        ax1.plot(distArr[:,4], 'r', label="Clust softmax")
        ax1.plot(distArr[:,6], 'cyan', label="Detect conv1")
        ax1.plot(distArr[:,8], 'blue', label="Detect conv2")
        ax1.plot(distArr[:,10], 'm', label="Detect softmax")

        plot.legend()
        plt.grid()
        plot.tight_layout()
        plt.savefig('../vis/{}/net{}_weightsDistances.png'.format(args.exp_id,i))

def activationSparsity(args):

    #Count the number of net in the experiment
    netNumber = len(glob.glob("../nets/{}/*.ini".format(args.exp_id)))

    #Get and sort the experiment file
    weigFiles = sortExperiFiles("../nets/"+args.exp_id+"/clustDetectNet*_epoch*".format(args.exp_id),netNumber)
    paramDictPaths = sorted(glob.glob("../nets/"+str(args.exp_id)+"/*.ini"))

    config = configparser.ConfigParser()

    _,test_loader = dataLoader.loadData(args.dataset,args.batch_size,args.test_batch_size)

    #Assuming the all the net in the exp have the same dataset
    #and the same value for the boolean parameter inweig
    config.read(paramDictPaths[0])
    dataset = config['default']["dataset"]
    inweig = (config['default']["inweig"] == 'True')
    clust = int(config['default']["clust"])

    #Plotting the loss across epoch and nets
    plotHist = plt.figure(1,figsize=(8,5))
    ax1 = plotHist.add_subplot(111)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    plt.xlabel('Epoch')
    plt.ylabel('Sparsity')
    handlesInp = []
    handlesConv1 = []
    handlesConv2 = []

    #cmap = cm.get_cmap(name='rainbow')
    colors = cm.rainbow(np.linspace(0, 1, len(weigFiles)))

    for i in range(len(weigFiles)):

        print("Net",i)
        #Reading general parameters
        config.read(paramDictPaths[i])
        paramDict = config['default']

        #check if net parameter are in the config file
        #if they are not : using the default ones
        if not 'biasclu' in config['default']:
            config.read("clust.config")

        config['default']["runCuda"] = str(args.cuda)

        paramNamespace = Bunch(config['default'])

        net = netBuilder.netMaker(paramNamespace)
        net.eval()

        sparsInpMean = np.empty((len(weigFiles[0])))
        sparsConv1Mean = np.empty((len(weigFiles[0])))
        sparsConv2Mean = np.empty((len(weigFiles[0])))

        for j in range(len(weigFiles[0])):

            net.load_state_dict(torch.load(weigFiles[i,j]))

            sparsInpMean[j] = 0
            sparsConv1Mean[j] = 0
            sparsConv2Mean[j] = 0

            for data, origTarget in test_loader:

                output,actArr = net(data)
                cluDis = net.cluDis
                clusts = actArr[2]
                maps = actArr[-3]
                summed_maps = actArr[-2]

                sparsInpMean[j] += computeSparsity(actArr[3]).mean()*len(data)/len(test_loader.dataset)
                sparsConv1Mean[j] += computeSparsity(actArr[4]).mean()*len(data)/len(test_loader.dataset)
                sparsConv2Mean[j] += computeSparsity(actArr[5]).mean()*len(data)/len(test_loader.dataset)

        label = ''.join((str(param)+"="+str(paramDict[param]+",")) for param in args.spar)

        handlesInp += ax1.plot(sparsInpMean, label=label,color=colors[i])
        handlesConv1 += ax1.plot(sparsConv1Mean, label=label,color=colors[i], dashes = [6,2])
        handlesConv2 += ax1.plot(sparsConv2Mean, label=label,color=colors[i], dashes = [2,2])

        ax1.set_ylim([0, 1])

        legInp = plotHist.legend(handles=handlesInp, loc='upper right' ,title="Input")
        legConv1 = plotHist.legend(handles=handlesConv1, loc='center right' ,title="Conv1")
        legConv2 = plotHist.legend(handles=handlesConv2, loc='lower right' ,title="Conv2")

        plotHist.gca().add_artist(legInp)
        plotHist.gca().add_artist(legConv1)
        plotHist.gca().add_artist(legConv2)

        plt.grid()
        plt.savefig('../vis/{}/histo.pdf'.format(args.exp_id))

def accuracyEvolution(args):

    config = configparser.ConfigParser()

    #The params file path. Used to get the parameters which is changing
    paramDictPaths = sorted(glob.glob("../nets/"+str(args.exp_id)+"/*.ini"))
    def loadAndGetParam(filePath,param):
        config.read(filePath)
        return config['default'][param]

    #Count the number of net in the experiment1
    netNumber = len(paramDictPaths)

    #Get and sort the experiment file
    if args.train:
        scorFiles = sortExperiFiles("../results/"+args.exp_id+"/*epoch*_train.csv",netNumber,args.nets_to_remove)
    else:
        scorFiles = sortExperiFiles("../results/"+args.exp_id+"/*epoch*[0-9].csv",netNumber,args.nets_to_remove)

    try:

        #Sorting the nets depending on the first varying parameter value (if this parameter is a number)
        config.read(paramDictPaths[0])

        #If the conversion raise an error then the first param is not a number and the net won't be sorted
        float(config['default'][args.acc_evol[0]])

        print("Sorting files depending on varying param value")
        params = []
        for i in range(len(paramDictPaths)):
            config.read(paramDictPaths[i])
            params.append(config['default'][args.acc_evol[0]])

        zipped = sorted(zip(params,scorFiles,paramDictPaths), key=lambda pair: float(pair[0]))

        scorFiles = np.array([x for _,x,_ in zipped])
        paramDictPaths = np.array([x for _,_,x in zipped])

    except ValueError:
        print("Not sorting files")

    #Two figures are plot : one with accuracy metrics, one with total variation of the accuracy metrics
    plotSco,plotTV = plt.figure(1,figsize=(8,5)),plt.figure(3,figsize=(8,5))

    ax1,ax3 = plotSco.add_subplot(111),plotTV.add_subplot(111)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.55, box.height])
    ax3.set_position([box.x0, box.y0, box.width * 0.55, box.height])

    handlesAcc,handlesAccClu,handlesAccNoDe,handlesAccDe,handlesTV = [[] for i in range(5)]

    colors = cm.rainbow(np.linspace(0, 1, len(scorFiles)))

    #These arrays will store the value of each metric at each epochs
    acc,clu,det,nodet = [np.zeros((len(scorFiles),len(scorFiles[0]))) for i in range(4)]

    #These arrays will store the total variation (TV) of each curve
    tvAcc,tvNoDet,tvDet,tvClu = [np.zeros((len(scorFiles))) for i in range(4)]

    labelList = []

    for i in range(len(scorFiles)):
        print("Net ",i)

        config.read(paramDictPaths[i])
        paramDict = config['default']

        for j in range(len(scorFiles[i])):
            csv = genfromtxt(scorFiles[i,j].decode("utf-8"), delimiter=',')
            fullTarget,binaryTarget,output,clusters = csv[:,0],csv[:,1],csv[:,2:4], csv[:,4:4+int(paramDict['clust'])]

            #The accuracy
            acc[i,j] = computeAcc(output,binaryTarget)

            if paramDict['full_clust'] == "True" or args.full_clust:
                clu[i,j] = computeClusAcc(clusters,fullTarget,int(paramDict['seed']),int(paramDict['clust']),args.exp_id,i,j,full_clust=True)
            else:
                #The clustering accuracy
                nodet[i,j],det[i,j] = computeClusAcc(clusters,fullTarget,int(paramDict['seed']),int(paramDict['clust']),args.exp_id,i,j,full_clust=False)

        label = ''.join((str(param)+"="+str(paramDict[param])+",") for param in args.acc_evol)
        labelList.append(label)
        #Ploting the curves and computing the total variation
        handlesAcc += ax1.plot(acc[i,:], label=label,color=colors[i],dashes = [6,2])
        tvAcc[i] = np.abs(acc[i,:-1]-acc[i,1:]).sum()
        if paramDict['full_clust'] == "True" or args.full_clust:
            handlesAccClu += ax1.plot(clu[i,:], label=label,color=colors[i])
            tvClu[i] = np.abs(clu[i,:-1]-clu[i,1:]).sum()
        else:
            handlesAccNoDe += ax1.plot(nodet[i,:], label=label,color=colors[i])
            handlesAccDe += ax1.plot(det[i,:], label=label,color=colors[i],dashes = [1,1])
            tvNoDet[i] = np.abs(nodet[i,:-1]-nodet[i,1:]).sum()
            tvDet[i] = np.abs(det[i,:-1]-det[i,1:]).sum()

    #Ploting the total variation histogram
    width=0.9
    if paramDict['full_clust'] == "True" or args.full_clust:
        curveDict = {0:tvAcc,1:tvClu}
        keys= ["tvAcc","tvClu"]
    else:
        curveDict = {0:tvAcc,1:tvNoDet,2:tvDet}
        TVkeys= ["tvAcc","tvNoDet","tvDet"]
    shift = (1.0*width)/len(curveDict.keys())
    defaultPosition = np.arange(len(curveDict[0]))-shift
    for i in range(len(curveDict.keys())):
        position = defaultPosition+(i*shift)
        handlesTV += ax3.bar(position, curveDict[i], width=(1.0*width)/len(curveDict.keys()))

    #Setting ylim
    ax1.set_ylim([0, 1])

    #Adding the legend
    legAcc = plotSco.legend(labels=labelList,handles=handlesAcc, loc='upper right' ,title="Accuracy",prop={'size': 6})

    if paramDict['full_clust'] == "True" or args.full_clust:
        legAccClu = plotSco.legend(labels=labelList,handles=handlesAccClu, loc='center right' ,title="Clustering",prop={'size': 6})
    else:
        legAccCluNoDe = plotSco.legend(labels=labelList,handles=handlesAccNoDe, loc='center right' ,title="Negative class clustering",prop={'size': 6})
        legAccCluDe = plotSco.legend(labels=labelList,handles=handlesAccDe, loc='lower right' ,title="Positive class clustering",prop={'size': 6})

    legTV = plotTV.legend(handlesTV,TVkeys, loc='upper right' ,title="Total variation")

    plotSco.gca().add_artist(legAcc)

    if paramDict['full_clust'] == "True" or args.full_clust:
        plotSco.gca().add_artist(legAccClu)
    else:
        plotSco.gca().add_artist(legAccCluNoDe)
        plotSco.gca().add_artist(legAccCluDe)

    plotTV.gca().add_artist(legTV)

    #Saving each figure
    plt.figure(1)
    plt.grid()
    plt.xlabel('Epoch')
    if args.train:
        plt.savefig('../vis/{}/{}_trainError.png'.format(args.exp_id,args.exp_id))
    else:
        plt.savefig('../vis/{}/{}_testError.png'.format(args.exp_id,args.exp_id))

    plt.figure(3)
    plt.grid()
    plt.xlabel('Net')
    if args.train:
        plt.savefig('../vis/{}/tv_train.png'.format(args.exp_id))
    else:
        plt.savefig('../vis/{}/tv_test.png'.format(args.exp_id))

def removeNets(scorFiles,netIdsToTemove):

    for netId in netIdsToTemove:
        scorFiles = list(filter(lambda x: x.find("net{}_".format(netId)) == -1,scorFiles))

    return scorFiles

def failuresCases(args):

    _,test_loader = dataLoader.loadData(dataset=args.dataset,batch_size=args.batch_size,test_batch_size=1,cuda=False)

    #Count the number of net in the experiment1
    netNumber = len(glob.glob("../nets/{}/*.ini".format(args.exp_id)))

    #Get and sort the experiment file
    if args.train:
        scorFiles = sortExperiFiles("../results/"+args.exp_id+"/*epoch*_train.csv",netNumber)
    else:
        scorFiles = sortExperiFiles("../results/"+args.exp_id+"/*epoch*[0-9].csv",netNumber)

    imgCounter = 0

    netId=args.failurecases[0]
    epoch=args.failurecases[1]

    for data, origTarget in test_loader:

        if imgCounter%10 == 0:
            print("Img ",imgCounter)

        if not (os.path.exists("../vis/{}/net{}".format(args.exp_id,netId))):
            os.makedirs("../vis/{}/net{}".format(args.exp_id,netId))

        if not os.path.exists("../vis/{}/net{}/epoch{}".format(args.exp_id,netId,epoch)):
            os.makedirs("../vis/{}/net{}/epoch{}".format(args.exp_id,netId,epoch))

        csv = genfromtxt(scorFiles[netId,epoch], delimiter=',')
        output = csv[imgCounter,2:4]
        binaryTarget = csv[imgCounter,1]
        fullTarget = csv[imgCounter,0]
        clusters = csv[imgCounter,4:9]

        if np.argmax(output) != binaryTarget:

            vis.writeImg("../vis/{}/net{}/epoch{}/{}.jpg".format(args.exp_id,netId,epoch,imgCounter),data[0][0].detach().numpy())

        imgCounter += 1

def meanClusterDistribEvol(args):

    netNumber = len(glob.glob("../nets/{}/*.ini".format(args.exp_id)))

    #Get and sort the experiment file
    if args.train:
        scorFiles = sortExperiFiles("../results/"+args.exp_id+"/*epoch*_train.csv",netNumber)
    else:
        scorFiles = sortExperiFiles("../results/"+args.exp_id+"/*epoch*[0-9].csv",netNumber)

    ind = np.arange(len(scorFiles[0]))
    width=1

    for i in range(len(scorFiles)):
        print("Net ",i)

        clusterMeansList = []
        for j in range(len(scorFiles[i])):
            csv = genfromtxt(scorFiles[i,j], delimiter=',')
            #Sort activation
            clusters = np.sort(csv[:,4:9])

            #Compute mean of sorted activation
            #The first value of clustersMean represent the mean of the larger value among the 5 outputs of the clustNet
            #The second value represent the second most important value, etc.
            clusterMeansList.append(np.mean(clusters,axis=0))

        clusterMeansList = np.array(clusterMeansList)

        plt.figure(i)
        barList = [plt.bar(ind, clusterMeansList[:,0], width)]

        cumSum = clusterMeansList[:,0]
        for j in range(1,len(clusterMeansList[0,:])):
            barList += plt.bar(ind,clusterMeansList[:,j], width,bottom=cumSum)
            cumSum += clusterMeansList[:,j]

        plt.ylabel('Activation')
        plt.title('Mean activation across epochs')

        if args.train:
            plt.savefig('../vis/{}/net{}_hist.pdf'.format(args.exp_id,i))
        else:
            plt.savefig('../vis/{}/net{}_hist_train.pdf'.format(args.exp_id,i))

def featureMapVariance(args):

    torch.manual_seed(args.seed)

    _,test_loader = dataLoader.loadData(dataset=args.dataset,batch_size=args.batch_size,test_batch_size=args.test_batch_size,cuda=False)

    netId=args.feat_map_var[0]
    layNb=args.feat_map_var[1]

    #Get and sort the experiment file
    weigFiles = sortExperiFiles("../nets/"+args.exp_id+"/clustDetectNet"+str(netId)+"_epoch*".format(args.exp_id),netNumber=1)
    paramDictPath = "../nets/"+str(args.exp_id)+"/clustDetectNet"+str(netId)+".ini"

    #Getting the dataset and the boolean parameter inweig
    #Assuming the all the net in the exp have the same dataset
    #and the same value for the boolean parameter inweig
    config = configparser.ConfigParser()
    config.read(paramDictPath)

    batch_nb = len(test_loader.dataset)//args.test_batch_size

    #Updating args with the argument in the config file
    argsDict = vars(args)
    for key in config['default']:

        if key in argsDict:
            if not argsDict[key] is None:
                cast_f = type(argsDict[key])

                if cast_f is bool:
                    cast_f = lambda x:True if x == "True" else False

                if config['default'][key][0] == "[" :
                    values = config['default'][key].replace("[","").replace("]","").split(" ")
                    argsDict[key] = [cast_f(value.replace(",","")) for value in values]
                else:
                    argsDict[key] = cast_f(config['default'][key])

    args = Bunch(argsDict)

    net = netBuilder.netMaker(args)
    net.eval()

    imgCounter = 0

    #Getting the size of feature map at the desired layer
    img = test_loader.dataset[0][0]
    imgSize = net(img[None,:,:,:])[1][-3].size(-1)

    plt.figure()
    epoch_count = 0
    colors = cm.rainbow(np.linspace(0, 1, args.clust))

    for weigFile in weigFiles[0]:
        epoch_count +=1
        print("Epoch",epoch_count)

        net.load_state_dict(torch.load(weigFile))

        batch_count = 1
        outputComputed = False

        for i in range(args.clust):
            open("feature_map_{}_pos_tmp.csv".format(i),'w')
            open("feature_map_{}_neg_tmp.csv".format(i),'w')

        clusDisSum = torch.zeros(args.clust)

        for data, origTarget in test_loader:

            target = mnist.merge(origTarget)

            if batch_count%(batch_nb//10) ==0:
                print("\tbatch",batch_count,"on",batch_nb)
            batch_count +=1

            _,actArr = net(data)

            act = actArr[-3].view(args.clust,-1,imgSize,imgSize)

            for i in range(len(act)):

                mapsPos = mnist.masked_index(act[i],0,(target != 0).long())
                mapsNeg = mnist.masked_index(act[i],0,((1-target) != 0).long())

                if mapsPos.size(0) != 0:
                    nonEmptyPos = mnist.masked_index(mapsPos,0,(mapsPos.sum(dim=1).sum(dim=1) != 0).long())
                    writeMap(nonEmptyPos,"feature_map_{}_pos_tmp.csv".format(i))

                if mapsNeg.size(0) != 0:
                    nonEmptyNeg = mnist.masked_index(mapsNeg,0,(mapsNeg.sum(dim=1).sum(dim=1) != 0).long())
                    writeMap(nonEmptyNeg,"feature_map_{}_neg_tmp.csv".format(i))

        plotVariance("pos",args.clust,epoch_count,colors,netId,layNb,args.exp_id)
        plotVariance("neg",args.clust,epoch_count,colors,netId,layNb,args.exp_id)

def hiddenRepTSNE(args):

    #Setting the size and the number of channel depending on the dataset
    if args.dataset == "MNIST":
        inSize = 28
        inChan = 1
    elif args.dataset == "CIFAR10":
        inSize = 32
        inChan = 3
    else:
        raise("netMaker: Unknown Dataset")

    net = netBuilder.CNN(inSize=inSize,inChan=inChan,chan=int(args.encchan),avPool=False,nbLay=int(args.encnblay),\
                      ker=int(args.encker),maxPl1=int(args.encmaxpl1),maxPl2=int(args.encmaxpl2),applyDropout2D=0,nbOut=0,\
                      applyLogSoftmax=False,nbDenseLay=0,sizeDenseLay=0)

    if args.cuda:
        net.cuda()

    net.setWeights(torch.load(args.tsne[0]),cuda=args.cuda,noise_init=0)

    train_loader,test_loader = dataLoader.loadData(args.dataset,int(args.batch_size),int(args.test_batch_size),args.cuda,int(args.num_workers))
    loader = train_loader if args.train else test_loader

    #Choosing a random batch of images among the dataset
    data,target = next(iter(loader))
    if args.cuda:
        data = data.cuda()
        target = target.cuda()

    #Computes the hidden representation of the batch of images
    repre,_,_ = net.convFeat(data)

    colors = cm.rainbow(np.linspace(0, 1, 10))

    #visualisation of the transformed data
    repre = repre.view(data.size(0),-1).cpu().detach().numpy()
    repre_emb = TSNE(n_components=2,init='pca',learning_rate=20).fit_transform(repre)
    plotEmb(repre_emb,target,"../vis/{}/{}_tsne.png".format(args.exp_id,args.ind_id),colors)

    ##Visualization of the raw data
    repre = data.view(data.size(0),-1).cpu().detach().numpy()
    repre_emb = TSNE(n_components=2,init='pca',learning_rate=20).fit_transform(repre)
    plotEmb(repre_emb,target,"../vis/{}/{}_tsne_raw.png".format(args.exp_id,args.ind_id),colors)

def sortExperiFiles(regex,netNumber,netsToRemove):
    '''Sort files on their net id and after on their epoch number

    Can sort results files (csv files in the \'results\' folder) or weight files (in the \'nets\' folder)

    Args:
        regex (string): the regex indicating which files has to be sorted.
            Example : \'../nets/clustNet_capacity/clustDetectNet*_epoch*\' will match will a the weight files from the experience \'clustNet_capacity\'.
        netNumber (int): the total number of networks in the experiment
        netsToRemove (list): the list of net id not to plot
    Returns:
        (numpy.ndarray): the files sorted in a 2D array. The rows corresponds to nets ID and columns corresponds to epoch number
    '''

    #Getting the weight of all the net at all the epochs
    files = glob.glob(regex)

    #Remove the nets not to plot
    files = removeNets(files,netsToRemove)

    #Sorting files depending on which net they correspond
    files = sorted(files,key=findFirstNumber)

    #print(files)

    maxLen = 0
    for i in range(len(files)):
        if len(files[i])> maxLen:
            maxLen = len(files[i])
    #Separating the files corresponding to each net
    files = np.array(files,dtype='S{}'.format(maxLen)).reshape((netNumber-len(netsToRemove),-1))

    #Sorting every line of the array depending on the epoch number
    kwargs = {'key': findNumbers}
    files = np.apply_along_axis(sorted,1,files,**kwargs)
    print(files)
    return files

def findNumbers(x):
    '''Extracts the numbers of a string and returns them as an integer'''

    return int((''.join(xi for xi in str(x) if xi.isdigit())))

def findFirstNumber(x):
    '''Extract the first sequence of digit of a string

    Example: passing the string \'clustDetectNet18_epoch45\' will returns the integer 18/

    '''

    i=0
    basename = os.path.basename(x)
    startReached = False
    endReached = False
    res = ''

    while i<len(basename):
        if basename[i].isdigit() and not endReached:
            if not startReached:
                startReached = True
            res += basename[i]
        elif startReached:
            endReached = True

        i+=1
    return int(res)

def plotEmb(data_emb,label,filepath,colors):
    ''' Plot a list of 2-dimensional points colored depending on their label

    Args:
        data_emb (numpy.ndarray): list of 2-dimensional points to plot
        label (torch.autograd.variable.Variable): the list of label to use to color each points. Must be as long as data_emb
        filepath (string): the path of the output file
        colors (list): the list of colors to use for each class
    '''

    plot = plt.figure()
    ax1 = plot.add_subplot(111)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width*0.8 , box.height])

    handles = []
    handlesLabel = []
    for i in range(10):
        points = data_emb[label.cpu().detach().numpy() == i]
        handlesLabel.append(str(i))
        handles.append(ax1.scatter(points[:,0],points[:,1],c=colors[i]))
    leg = plot.legend(handles,handlesLabel)
    plot.gca().add_artist(leg)

    plt.savefig(filepath)

def plotVariance(mode,clusts,epoch_count,colors,netId,layNb,exp_id):
        '''Read csv files containing the one-channel feature maps and plot pixels variance with a multi-bar histogram

        Args:
            mode (str): a string indicating the feature map to be read should be feature maps of positive example or negative example.
                Should be 'neg' or 'pos'.
            epoch_count (int): the epoch number
            colors (numpy.ndarray): an array of color produced by the matplotlib.cm.rainbow() method. Used to assign a color to each bar of a multi-bar
            netId (int): the net id
            layNb (int): the layer number
            exp_id (str): the experience id

        '''

        average_stds= []
        barList = []
        width=0.9
        barSum = 0
        for i in range(clusts):

            csv = np.genfromtxt("feature_map_{}_{}_tmp.csv".format(i,mode),delimiter=',')
            if csv.ndim > 1:

                #Computing the standard deviation of each column (i.e. of each pixel in a feature map
                std = np.std(csv,axis=0)
                #Computing the mean of standard deviation across pixels
                std = std.mean()

                #Plot the bar with heigth corresponding to std
                average_stds.append(std)
                barList += plt.bar(epoch_count,average_stds, width,bottom=barSum,color=colors[i])
                barSum += std

        plt.xlabel("Epochs")
        plt.ylabel('Standard deviation')
        plt.title('Standard deviation across epochs')
        plt.savefig('../vis/{}/net{}_lay{}_{}_var.pdf'.format(exp_id,netId,layNb,mode))

def writeMap(maps,filename):
    '''writes one-channel feature maps in a csv file

    The tensor passed as input will be reshaped to a 2D tensor array. So the rows correspond to the image in the batch
        and the columns correspond to pixels

    Args:
        maps (torch.tensor): a 3D tensor array with shape (image_number,width,heigth).
        filename (string): the name of the csv file
    '''

    if maps.size(0) != 0:
        maps = maps.view(maps.size(0),-1)

        with open(filename,'a') as text_file:
            maps = maps.detach().numpy()
            for i in range(len(maps)):
                for j in range(len(maps[0])):
                    print(str(maps[i,j]),file=text_file,end='')

                    #Writing a comma if the end of line has not been reached yet
                    if j !=  len(maps[0]) - 1:
                        print(",",file=text_file,end='')
                    else:
                        print("",file=text_file)

def computeSparsity(arr):
    '''Compute the proportion of element equals to zero in a tensor array'''

    #The number of elements equal to zero
    zerosNb = (arr.view(arr.size()[0],-1) == 0).sum(dim=1).float()
    #The total number of elements
    elemNb = arr.view(arr.size()[0],-1).size()[1]
    sparsity =  zerosNb/elemNb
    return sparsity

def computeAcc(output,binaryTarget):
    '''Compute the accuracy of a batch of prediction

    output (numpy.ndarray): a 2D array with all the predictions of the batch. The rows corresponds to each image in the batch
        The first column indicates the score of the positive class and the other column indicates the score of the negative class
    binaryTarget (numpy.ndarray): a 1D array with the true class indexs of the batch images.
    Returns:
        the accuracy of the predictions in the batch
    '''

    output = torch.tensor(output)
    binaryTarget = torch.LongTensor(binaryTarget)
    pred = torch.tensor(output).data.max(1, keepdim=True)[1]
    correct = pred.eq(binaryTarget.data.view_as(pred)).long().cpu().sum()
    return float(correct)/len(binaryTarget)

def computeClusAcc(clusters,fullTarget,seed,clusterNb,exp_id,i,j,full_clust=False):
    """
    Attribute a cluster to a class, compute the accuracy and plot the confusion matrix
    :param clusters: the list of the distributions given by the clust net for each test image (np array like)
    :param fullTarget: the list of targets for each test image (not binary target) (np array like)
    :param seed: the seed used to determine the class to find
    :param clusterNb: the number of cluster used
    :return: the cluster accuracies for the class to detect and the classes not to detect
    """

    clustPred = torch.tensor(clusters).data.max(1, keepdim=True)[1]
    np.random.seed(seed)
    classes = list(np.arange(10))
    np.random.shuffle(classes)
    classToFind =  classes[0: clusterNb]
    confMat = trainVal.computeConfMat(clustPred,fullTarget, heatmapFileName=None,classToFind = classToFind,exp_id=exp_id,nbClusts=clusterNb)

    #print(confMat)

    def assignAndPlot(confMat,exp_id,i,j,classes,mode):

        #Assigning the clusters to half of the classes
        assign = sp.optimize.linear_sum_assignment(-confMat)
        mat = permutate(confMat,assign)
        #computing the clustering accuracy for the classes not to detect, and plotting the permuted heatmap
        acc = mat.trace()/mat.sum()

        plotHeatMaps(mat,'../vis/'+str(exp_id)+"/"+"perm_{}_{}_{}.png".format(mode,i,j),len(classes))

        return acc

    if full_clust:
        acc = assignAndPlot(confMat,exp_id,i,j,classes,'full')
        return acc
    else:
        noDetAcc = assignAndPlot(confMat[:,5:],exp_id,i,j,classes,'det')
        detAcc = assignAndPlot(confMat[:,:5],exp_id,i,j,classes,'no_det')
        return noDetAcc,detAcc

def plotHeatMaps(confMat,fileName,nbClasses):
    '''Produce a heatmap visualisation of a square confusion matrix

    Args:
        confMat (numpy.ndarray): a 2D numpy array
        fileName (string): the file name of the image to write
    '''

    nbClasses = confMat.shape[0]

    heatmap = plt.figure()
    ax1 = heatmap.add_subplot(111)
    plt.xticks(np.arange(nbClasses),[i for i in range(nbClasses)])
    plt.imshow(confMat, cmap='gray', interpolation='nearest')
    plt.savefig(fileName)
    plt.close()

def permutate(mat, perm):
    """
    Permutate the rows of a matrix according to a permutation
    :param mat: a matrix (np array)
    :param perm: a permutation. An array of row indices and one of corresponding column indices giving the optimal assignment. (tuple)
    :return: matrix permuted
    """

    tmp=np.empty_like(mat)
    for p in range(len(perm[0])):
        tmp[perm[1][p]] = mat[perm[0][p]]
    return tmp

class Bunch(object):
  '''Convert a dictionnary into a namespace object'''
  def __init__(self, dict):
    self.__dict__.update(dict)

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--weig',action='store_true',help='Plot distance between the weights and their previous version at each epoch. \
                        The weights file name should be like "clustDetectNet7_epoch12"')

    argreader.parser.add_argument('--spar', nargs='+',type=str, metavar='HIST',help='Plot the mean sparsity of activation. The values after "--spar" \
                        is the list of parameters which are varying across the different nets. The score file must contain the targets, the prediction and cluster scores.')

    argreader.parser.add_argument('--acc_evol', nargs='+',type=str, metavar='SCOR',
                        help='Plot the error, accuracy and clustering score across the training epochs for all nets in the experiment. Also plot \
                        the total variation of these curves. The values after "--acc_evol" is the list of parameters which are varying among the \
                        different nets. Some nets can be not plotted using the --nets_to_remove arg.')

    argreader.parser.add_argument('--nets_to_remove', nargs='+',default=[],type=int, metavar='ID',
                        help='The list of net IDs not to plot when using the --acc_evol arg.')

    argreader.parser.add_argument('--failurecases',type=int, nargs=2,metavar='SCOR',
                        help='To write in a folder the test images that are misclassified by the nets at every epoch. '\
                        'The first parameter after this argument is the net index and the second one is the epoch number')

    argreader.parser.add_argument('--cludis',type=int, nargs=2,metavar='SCOR',
                        help='To plot the average cluster distribution across epochs for all nets in an experiment')

    argreader.parser.add_argument('--feat_map_var',type=int, nargs=2,metavar='ENTR_EPOCH',
                        help='For each detecting network, plot the feature map average variance across epochs. Positive and negative inputs are also \
                        separated. The first argument is the net id and the second is the layer number')

    argreader.parser.add_argument('--tsne',type=str,nargs=2,metavar="PATH",
                        help='To plot t-sne on the reprenstations produced by an encoder. The fisrt argument must be the path to an CNN model and the second\
                        is the number of samples to use.')

    argreader.parser.add_argument('--train',action='store_true',help='To do all computation on the train set')

    #Reading the comand line arg
    argreader.getRemainingArgs()
    #Getting the args from command line and config file
    args = argreader.args

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.weig :
        weigthTrajectoryLength(args)

    if args.spar:
        activationSparsity(args)

    if args.acc_evol:
        accuracyEvolution(args)

    if args.failurecases:
        failuresCases(args)

    if args.cludis:
        meanClusterDistribEvol(args)

    if args.feat_map_var:
        featureMapVariance(args)

    if args.tsne:
        hiddenRepTSNE(args)

if __name__ == '__main__':
    main()
