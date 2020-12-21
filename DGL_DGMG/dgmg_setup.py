import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
from DGL_DGMG import common


def weights_init(m):
    '''
    Code from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    #FC layers
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        try:
            init.normal_(m.bias.data)
        except:
            pass
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, common.MLP):
        #For each FC layer in the MLP
        for layer in m.linears:
            init.xavier_normal_(layer.weight.data)
            try:
                init.normal_(layer.bias.data)
            except:
                pass

def dgmg_message_weight_init(m):
    """
    This is similar as the function above where we initialize linear layers from a normal distribution with std
    1./10 as suggested by the author. This should only be used for the message passing functions, i.e. fe's in the
    paper.
    """
    def _weight_init(m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight.data, std=1./10)
            init.normal_(m.bias.data, std=1./10)
        else:
            raise ValueError('Expected the input to be of type nn.Linear!, got {}'.format(type(m)))


    if isinstance(m, nn.Linear):
        m.apply(_weight_init)
    elif isinstance(m, common.MLP):
        for layer in m.linears:
            layer.apply(_weight_init)
    elif isinstance(m, nn.ModuleList):
        for layer in m:
            if isinstance(layer, common.MLP):
                for sublayer in layer.linears:
                    if isinstance(sublayer, nn.Linear):
                        sublayer.apply(_weight_init)
            elif not isinstance(layer, nn.BatchNorm1d):
                layer.apply(_weight_init)
