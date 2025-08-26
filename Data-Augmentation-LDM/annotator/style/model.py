import torch.nn as nn
import collections as c

whichChannel = "2d"
N_FILTERS = 64 #4096
hor_filter = 11
N_SAMPLES = 1024 #time bins
N_FREQ = 64 #freq bins

if whichChannel == "2d":
    IN_CHANNELS = 1
elif whichChannel == "freq":
    IN_CHANNELS = N_FREQ
elif whichChannel == "time":
    IN_CHANNELS = N_SAMPLES

class style_net(nn.Module):
    """Here create the network you want to use by adding/removing layers in nn.Sequential"""
    def __init__(self):
        super(style_net, self).__init__()
        self.layers = nn.Sequential(c.OrderedDict([
                            ('conv1',nn.Conv2d(IN_CHANNELS,N_FILTERS,kernel_size=(1,hor_filter), padding=(0,5), bias=False)),
#                             ('batch1',nn.BatchNorm2d(100)),
                            ('relu1',nn.ReLU()),
                            ('max1', nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))),
                            ('conv2',nn.Conv2d(N_FILTERS,N_FILTERS//2,kernel_size=(1,hor_filter), padding=(0,5),bias=False)),
#                             ('batch2',nn.BatchNorm2d(100)),
                            ('relu2',nn.ReLU()),
                            ('max2', nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))),
                            ('conv3',nn.Conv2d(N_FILTERS//2,N_FILTERS//4,kernel_size=(1,hor_filter), padding=(0,5), bias=False)),
#                             ('batch3',nn.BatchNorm2d(100)),
                            ('relu3',nn.ReLU()),
                            ('max3', nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))),
                            ('flat',nn.Flatten()),
                            # ('flat',nn.Flatten(start_dim=0,end_dim=-1)),
                            ('linear',nn.Linear(16384,512))]))
        
    def forward(self,input):
        out = self.layers(input)
        return out