from ultis import extract_commit, reformat_commit_code
from main_DCNN import read_args
from padding import padding_commit
from ultis import mini_batches
import torch
from PatchNet_CNN import PatchNet
import os
import datetime
import numpy as np


def collect_batches(commits, params):
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = \
        padding_commit(commits=commits, params=params)
    print('Dictionary of commit message has size: %i' % (len(dict_msg)))
    print('Dictionary of commit code has size: %i' % (len(dict_code)))

    print(pad_msg.shape, pad_added_code.shape, pad_removed_code.shape, labels.shape)

    batches = mini_batches(X_msg=pad_msg, X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=labels,
                           mini_batch_size=params.batch_size)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda

    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    print("\nParameters:")
    for attr, value in sorted(params.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PatchNet(args=params)
    return batches, model


def commit_embedding(path, commits, params):
    batches, model = collect_batches(commits=commits, params=params)
    model.load_state_dict(torch.load(path))
    embedding_vectors, cnt = list(), 0
    for batch in batches:
        pad_msg, pad_added_code, pad_removed_code, labels = batch
        print(type(pad_msg))
        exit()
        if torch.cuda.is_available():
            pad_msg, pad_added_code, pad_removed_code, labels = torch.tensor(pad_msg).long(), torch.tensor(
                pad_added_code).long(), torch.tensor(pad_removed_code).long(), torch.tensor(labels).float()
        else:
            pad_msg, pad_added_code, pad_removed_code, labels = torch.tensor(pad_msg).cuda.long(), torch.tensor(
                pad_added_code).cuda.long(), torch.tensor(pad_removed_code).cuda.long(), torch.tensor(
                labels).cuda.float()
        predict = model.forward(pad_msg, pad_added_code, pad_removed_code)
        commits_vector = model.forward_commit_embeds(pad_msg, pad_added_code, pad_removed_code)

        print(type(predict))
        print(predict.shape)
        print(predict)
        print(commits_vector.shape)
        if torch.cuda.is_available():
            commits_vector = commits_vector.detach().numpy()
        else:
            commits_vector = commits_vector.cpu().detach().numpy()
        print(commits_vector.shape)
        print(type(commits_vector))
        print('cuda test')
        exit()
        cnt += 1


if __name__ == '__main__':
    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    # path_file = './data/newres_funcalls_words_jul28.out'
    path_file = './data/small_newres_funcalls_words_jul28.out'
    commits = extract_commit(path_file=path_file)
    commits = reformat_commit_code(commits=commits, num_file=1, num_hunk=8,
                                   num_loc=10, num_leng=120)
    date = '/2018-11-13_14-41-49/'
    path_model = './snapshot/2018-11-13_14-41-49/epoch_epochs_18.pt'
    commit_embedding(path=path_model, commits=commits, params=input_option)
