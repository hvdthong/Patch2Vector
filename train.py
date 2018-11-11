#  * This file is part of PatchNet, licensed under the terms of the GPL v2.
#  * See copyright.txt in the PatchNet source code for more information.
#  * The PatchNet source code can be obtained at
#  * https://github.com/hvdthong/PatchNetTool

from padding import padding_commit
import os
import datetime
from ultis import random_mini_batch, write_dict_file
import torch
from PatchNet_CNN import PatchNet

def running_train(batches, model, params):
    print('hello')
    exit()



def train_model(commits, params):
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = \
        padding_commit(commits=commits, params=params)
    print('Dictionary of commit message has size: %i' % (len(dict_msg)))
    print('Dictionary of commit code has size: %i' % (len(dict_code)))

    print(pad_msg.shape, pad_added_code.shape, pad_removed_code.shape, labels.shape)

    batches = random_mini_batch(X_msg=pad_msg, X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=labels,
                                mini_batch_size=params.batch_size)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda

    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.embed_msg, params.embed_code = len(dict_msg), len(dict_code)
    params.class_num = labels.shape[1]

    print("\nParameters:")
    for attr, value in sorted(params.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PatchNet(args=params)
    running_train(batches=batches, model=model, params=params)

    exit()
