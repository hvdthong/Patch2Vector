import torch.nn as nn


class PatchNet(nn.Module):
    def __init__(self, args):
        super(PatchNet, self).__init__()
        self.args = args

        V = args.msg_length
        D = args.embed_dim
        C = args.class_num
