import  torch.utils.data
from torchvision import datasets, transforms

def loadData(dataset,batch_size,test_batch_size,cuda=False,num_workers=1):
    ''' Build two dataloader

    Args:
        dataset (string): the name of the dataset. Can be \'MNIST\' or \'CIFAR10\'.
        batch_size (int): the batch length for training
        test_batch_size (int): the batch length for testing
        cuda (bool): whether or not to run computation on gpu
        num_workers (int): the number of workers for loading the data.
            Check pytorch documentation (torch.utils.data.DataLoader class) for more details
    Returns:
        train_loader (torch.utils.data.dataloader.DataLoader): the dataLoader for training
        test_loader (torch.utils.data.dataloader.DataLoader): the dataLoader for testing

    '''

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if cuda else {}

    if dataset == "MNIST":
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data/MNIST', train=True, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data/MNIST', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=test_batch_size, shuffle=False, **kwargs)

    elif dataset == "CIFAR10":
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data/', train=True, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data/', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError("Unknown dataset",dataset)

    return train_loader,test_loader

if __name__ == '__main__':

    train,_ = loadData("CIFAR10",1,1,cuda=False,num_workers=1)

    print(type(train))
