from main_DCNN import read_args
from ultis import load_file
from Jiang_baseline_data_preparation import load_Jiang_code_data
import os
import torch
from Jiang_baseline_data_padding import padding_commit_code, commit_msg_label
from ultis import mini_batches_topwords
import numpy as np
import datetime
from Jiang_baseline_data_PatchEmbedding import PatchEmbedding
import torch.nn as nn
from train_topwords import save
from Jiang_baseline_data_train import reshape_code
from load_embedding_topwords import commit_embedding


def collect_batches(commit_diff, commit_msg, params, padding_code_info):
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
    return batches, model


if __name__ == '__main__':
    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    # loading the commit code
    ##################################################################################
    ##################################################################################
    path_train_diff = './data/2017_ASE_Jiang/train.26208.diff'
    data_train_diff = load_Jiang_code_data(pfile=path_train_diff)
    path_test_diff = './data/2017_ASE_Jiang/test.3000.diff'
    data_test_diff = load_Jiang_code_data(pfile=path_test_diff)
    data_diff = data_train_diff + data_test_diff
    padding_code_info = (15, 40)  # max_line = 15; max_length = 40
    ##################################################################################
    ##################################################################################

    # loading the commit msg
    ##################################################################################
    ##################################################################################
    path_train_msg = './data/2017_ASE_Jiang/train.26208.msg'
    data_train_msg = load_file(path_file=path_train_msg)
    path_test_msg = './data/2017_ASE_Jiang/test.3000.msg'
    data_test_msg = load_file(path_file=path_test_msg)
    data_msg = data_train_msg + data_test_msg

    input_option.filter_sizes = [int(k) for k in input_option.filter_sizes.split(',')]
    # batches, model = collect_batches(commit_diff=data_diff[:500], commit_msg=data_msg[:500], params=input_option,
    #                                  padding_code_info=padding_code_info)
    batches, model = collect_batches(commit_diff=data_diff, commit_msg=data_msg, params=input_option,
                                     padding_code_info=padding_code_info)
    for epoch in range(input_option.start_epoch, input_option.end_epoch + 1):
        path_model = './snapshot/' + input_option.datetime + '/epoch_' + str(epoch) + '.pt'
        commit_embedding(path=path_model, batches=batches, model=model, params=input_option, nepoch=epoch)
