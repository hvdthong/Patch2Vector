import torch.nn as nn


class Double_CNN(nn.Module):
    def __init__(self, args):
        super(Double_CNN, self).__init__()
        self.args = args
