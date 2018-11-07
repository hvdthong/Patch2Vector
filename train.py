#  * This file is part of PatchNet, licensed under the terms of the GPL v2.
#  * See copyright.txt in the PatchNet source code for more information.
#  * The PatchNet source code can be obtained at
#  * https://github.com/hvdthong/PatchNetTool

from padding import padding_commit
import os
import datetime
from ultis import random_mini_batch, write_dict_file


def train_model(commits, params):
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = \
        padding_commit(commits=commits, params=params)
    print('Dictionary of commit message has size: %i' % (len(dict_msg)))
    print('Dictionary of commit code has size: %i' % (len(dict_code)))
    # print pad_msg.shape, pad_added_code.shape, pad_removed_code.shape, labels.shape

    print(pad_msg.shape)
    print(pad_added_code.shape)
    print(pad_removed_code.shape)
    print(labels.shape)
