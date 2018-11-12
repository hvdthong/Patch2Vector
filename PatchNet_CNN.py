import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class PatchNet(nn.Module):
    def __init__(self, args):
        super(PatchNet, self).__init__()
        self.args = args

        V_msg = args.embed_msg
        V_code = args.embed_code
        Dim = args.embedding_dim
        Class = args.class_num

        Ci = 1  # input of convolutional layer
        Co = args.num_filters  # output of convolutional layer
        Ks = args.filter_sizes  # kernel sizes

        # CNN for commit message
        self.embed_msg = nn.Embedding(V_msg, Dim)
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])

        # other information
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(3 * len(Ks) * Co, Class)

    def forward_msg(self, x, conv):
        x = self.embed_msg(x.cuda() if torch.cuda.is_available() else x)
        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)


    def forward(self, x):
        x_msg = self.forward_msg(x, self.convs_msg)
