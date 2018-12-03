import sys
import argparse
import configparser

def str2bool(v):
    '''Convert a string to a boolean value'''
    if v == 'True':
        return True
    elif v == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def str2FloatList(x):
    '''Convert a string to a list of float value'''
    if len(x.split(" ")) == 1:
        return float(x)
    else:
        return [float(elem) for elem in x.split(" ")]

class ArgReader():
    """
    This class build a namespace by reading arguments in both a config file
    and the command line.

    If an argument exists in both, the value in the command line overwrites
    the value in the config file

    This class mainly comes from :
    https://stackoverflow.com/questions/3609852/which-is-the-best-way-to-allow-configuration-options-be-overridden-at-the-comman
    Consulted the 18/11/2018

    """

    def __init__(self,argv):
        ''' Defines the arguments used in several scripts of this project.
        It reads them from a config file
        and also add the arguments found in command line.

        If an argument exists in both, the value in the command line overwrites
        the value in the config file
        '''

        # Do argv default this way, as doing it in the functional
        # declaration sets it at compile time.
        if argv is None:
            argv = sys.argv

        # Parse any conf_file specification
        # We make this parser with add_help=False so that
        # it doesn't parse -h and print help.
        conf_parser = argparse.ArgumentParser(
            description=__doc__, # printed with -h/--help
            # Don't mess with format of description
            formatter_class=argparse.RawDescriptionHelpFormatter,
            # Turn off help, so we print all options in response to -h
            add_help=False
            )
        conf_parser.add_argument("-c", "--conf_file",
                            help="Specify config file", metavar="FILE")
        args, self.remaining_argv = conf_parser.parse_known_args()

        defaults = {}

        if args.conf_file:
            config = configparser.SafeConfigParser()
            config.read([args.conf_file])
            defaults.update(dict(config.items("default")))

        # Parse rest of arguments
        # Don't suppress add_help here so it will handle -h
        self.parser = argparse.ArgumentParser(
            # Inherit options from config_parser
            parents=[conf_parser]
            )
        self.parser.set_defaults(**defaults)

        # Training settings
        #parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        self.parser.add_argument('--batch_size', type=int, metavar='N',
                            help='input batch size for training')
        self.parser.add_argument('--test_batch_size', type=int, metavar='N',
                            help='input batch size for testing')
        self.parser.add_argument('--epochs', type=int, metavar='N',
                            help='number of epochs to train')
        self.parser.add_argument('--lr', type=str2FloatList, nargs='+',metavar='LR',
                            help='learning rate (it can be a schedule : --lr 0.01 0.001 0.0001)')
        self.parser.add_argument('--lr_cl', type=str2FloatList, nargs='+',metavar='LR_CLUST',
                            help='learning rate of the clust net (it can be a schedule : --lr 0.01 0.001 0.0001)')
        self.parser.add_argument('--num_workers', type=int,metavar='NUMWORKERS',
                            help='the number of processes to load the data. num_workers equal 0 means that it’s \
                            the main process that will do the data loading when needed, num_workers equal 1 is\
                            the same as any n, but you’ll only have a single worker, so it might be slow')

        self.parser.add_argument('--momentum', type=float, metavar='M',
                            help='SGD momentum')
        self.parser.add_argument('--no_cuda', type=str2bool,metavar='NOCUDA',
                            help='disables CUDA training')
        self.parser.add_argument('--seed', type=int, metavar='S',
                            help='random seed')
        self.parser.add_argument('--log_interval', type=int, metavar='N',
                            help='how many batches to wait before logging training status')
        self.parser.add_argument('--ind_id', type=int, metavar='IND_ID',
                            help='the id of the individual')
        self.parser.add_argument('--exp_id', type=str, metavar='EXP_ID',
                            help='the id of the experience')
        self.parser.add_argument('--cl_to_find_size', type=int,metavar='N',
                            help='the size of the group of class to detect')
        self.parser.add_argument('--clust', type=int, metavar='N',
                            help='the number of cluster to do')
        self.parser.add_argument('--dataset', type=str, metavar='N',
                            help='the dataset to use')
        self.parser.add_argument('--full_class', type=str2bool, metavar='N',
                            help='indicates if the net has to detect or classify classes')
        self.parser.add_argument('--batch_period', type=int, metavar='N',
                            help='the period with which the two parts of the nets are training/not training. If set to 0, the nets are trained simulteneoulsy')

        self.parser.add_argument('--full_clust', type=str2bool, metavar='N',
                            help='Whether the net should be a full clustering network or not. Ignored if pretraining')

        self.parser.add_argument('--preproc', type=str, metavar='N',
                            help='The path to the network used to encode to image')

        self.parser.add_argument('--harddecision' ,action='store_true',
                            help='Whether to make a hard decision at the clustering level or not')
        self.parser.add_argument('--avpool' , type=str2bool, metavar='N',
                            help='Whether to apply average pooling on the last layer of the detectNets or not')
        self.parser.add_argument('--pretrain',type=str2bool, metavar='N', help='Set this to True to train only a detetect net')
        self.parser.add_argument('--pretrain_cae',type=str2bool, metavar='N', help='Set this to True to train a cae (used to initialize the clust net weights)')

        self.parser.add_argument('--decision_clu' , type=str, metavar='N',
                            help='The type of decision to make for the clustering network. Can be FS (full stochastic), DE (deterministic eval),\
                            FD (full deterministic), SD (soft decision) or GSF (gumbel-softmax) ')

        self.parser.add_argument('--temperature', type=float, metavar='M',
                            help='The temperature to use for gumbel_softmax. Used as start value during training. Ignored if another activation is used.')

        self.parser.add_argument('--clbn' , type=str2bool, metavar='N',
                            help='Whether to apply batch normalization on the clustering net or not')
        self.parser.add_argument('--clnblayers', type=int, metavar='N',
                            help='the nb of layers in the clustering net')
        self.parser.add_argument('--clker', type=int, metavar='N',
                            help='the kernel size of the conv layer of the clustering network (exept the last one)')
        self.parser.add_argument('--clchan', type=int, metavar='N',
                            help='the number of channel of the conv layer of the clustering network (exept the last one)')
        self.parser.add_argument('--clchan_out', type=int, metavar='N',
                            help='the number of channel of the last conv layer of the clustering network')
        self.parser.add_argument('--clmaxpoolsize', type=int, metavar='N',
                            help='the pooling size for the layers of the clustering network (except the last one)')
        self.parser.add_argument('--clmaxpoolsize_out', type=int, metavar='N',
                            help='the pooling size for the last conv layer of the clustering network')
        self.parser.add_argument('--cldrop', type=str2bool, metavar='N',
                            help='Whether to add or not 2D dropout to conv layer of the clustering net during training')
        self.parser.add_argument('--clnb_denselayers', type=int, metavar='N',
                            help='the number of dense layers for the clustering net')
        self.parser.add_argument('--clsize_denselayers', type=int, metavar='N',
                            help='the size of dense layers for the clustering net')

        self.parser.add_argument('--debn' , type=str2bool, metavar='N',
                            help='Whether to apply batch normalization on the detection net or not')
        self.parser.add_argument('--denblayers', type=int, metavar='N',
                            help='the nb of layers in the detecting net')
        self.parser.add_argument('--deker', type=int, metavar='N',
                            help='the kernel size of the layers of the detecting network (except the last one)')
        self.parser.add_argument('--dechan', type=int, metavar='N',
                            help='the number of channel of the layers (except the last one) of the detecting network')
        self.parser.add_argument('--dechan_out', type=int, metavar='N',
                            help='the number of channel of the last layer of the detecting network')
        self.parser.add_argument('--demaxpoolsize', type=int, metavar='N',
                            help='the pooling size for the first conv layer of the detecting network')
        self.parser.add_argument('--demaxpoolsize_out', type=int, metavar='N',
                            help='the pooling size for the last conv layer of the detecting network')
        self.parser.add_argument('--dedrop', type=str2bool, metavar='N',
                            help='Whether to add or not 2D dropout to conv layer of the detecting net during training')
        self.parser.add_argument('--denb_denselayers', type=int, metavar='N',
                            help='the number of dense layers for the dectecting nets')
        self.parser.add_argument('--desize_denselayers', type=int, metavar='N',
                            help='the size of dense layers for the detecting nets')

        self.parser.add_argument('--encchan', type=int, metavar='N',
                            help='the number of channel of the encoder net')
        self.parser.add_argument('--encnblay', type=int, metavar='N',
                            help='the number of layer in the encoder net')
        self.parser.add_argument('--encker', type=int, metavar='N',
                            help='the kernel size of the encoder net')
        self.parser.add_argument('--encmaxpl1', type=int, metavar='N',
                            help='the size of the first max pooling in the encoder net')
        self.parser.add_argument('--encmaxpl2', type=int, metavar='N',
                            help='the size of the second max pooling in the encoder net')
        self.parser.add_argument('--encdrop', type=str2bool, metavar='N',
                            help='whether or not to apply 2D dropout in the encoder net')
        self.parser.add_argument('--enchidd_repr_size', type=int, metavar='N',
                            help='The size of the representation produced by the encoder net')

        self.args = None

    def getRemainingArgs(self):
        ''' Reads the comand line arg'''

        self.args = self.parser.parse_args(self.remaining_argv)

    def writeConfigFile(self,filePath):
        """ Writes a config file containing all the arguments and their values"""

        config = configparser.SafeConfigParser()
        config.add_section('default')

        for k, v in  vars(self.args).items():
            config.set('default', k, str(v))

        with open(filePath, 'w') as f:
            config.write(f)
