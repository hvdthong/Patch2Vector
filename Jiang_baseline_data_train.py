import os
import torch
from Jiang_baseline_data_padding import padding_commit_code, commit_msg_label
from ultis import mini_batches_topwords
import numpy as np
import datetime
from PatchEmbedding_CNN import PatchEmbedding
from train_topwords import running_train


def reshape_code(data):
    ncommit, nline, nlength = data.shape[0], data.shape[1], data.shape[2]
    data = np.reshape(data, (ncommit, 1, nline, nlength))
    return data



def train_model(commit_diff, commit_msg, params, padding_code_info):
    max_line, max_length = padding_code_info[0], padding_code_info[1]
    pad_removed_code, pad_added_code, dict_code = padding_commit_code(data=commit_diff, max_line=max_line,
                                                                      max_length=max_length)
    pad_removed_code, pad_added_code = reshape_code(data=pad_removed_code), reshape_code(data=pad_added_code)
    print('Dictionary of commit code has size: %i' % (len(dict_code)))
    print('Shape of removed code and added code:', pad_removed_code.shape, pad_added_code.shape)

    labels, dict_msg = commit_msg_label(data=commit_msg)
    batches = mini_batches_topwords(X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=labels,
                                    mini_batch_size=params.batch_size)

    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    params.vocab_code = len(dict_code) + 1
    params.code_line = max_line
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PatchEmbedding(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    running_train(batches=batches, model=model, params=params)


