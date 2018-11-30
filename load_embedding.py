from ultis import extract_commit, reformat_commit_code
from main_DCNN import read_args
from padding import padding_commit
from ultis import mini_batches
import torch
from PatchNet_CNN import PatchNet
import os
import datetime
import numpy as np
import sys


def collect_batches(commits, params):
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = \
        padding_commit(commits=commits, params=params)
    print('Dictionary of commit message has size: %i' % (len(dict_msg)))
    print('Dictionary of commit code has size: %i' % (len(dict_code)))

    print(pad_msg.shape, pad_added_code.shape, pad_removed_code.shape, labels.shape)

    batches = mini_batches(X_msg=pad_msg, X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=labels,
                           mini_batch_size=params.batch_size)
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
    if torch.cuda.is_available():
        model = model.cuda()
    return batches, model


def commit_embedding(path, batches, model, params, nepoch):
    model.load_state_dict(torch.load(path))
    embedding_vectors, cnt = list(), 0
    print(path)
    for batch in batches:
        pad_msg, pad_added_code, pad_removed_code, labels = batch
        if torch.cuda.is_available():
            pad_msg, pad_added_code, pad_removed_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                pad_added_code).cuda(), torch.tensor(pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)
        else:
            pad_msg, pad_added_code, pad_removed_code, labels = torch.tensor(pad_msg).long(), torch.tensor(
                pad_added_code).long(), torch.tensor(pad_removed_code).long(), torch.tensor(labels).float()

        # predict = model.forward(pad_msg, pad_added_code, pad_removed_code)
        commits_vector = model.forward_commit_embeds(pad_msg, pad_added_code, pad_removed_code)

        if torch.cuda.is_available():
            commits_vector = commits_vector.cpu().detach().numpy()
        else:
            commits_vector = commits_vector.detach().numpy()

        if cnt == 0:
            embedding_vectors = commits_vector
        else:
            embedding_vectors = np.concatenate((embedding_vectors, commits_vector), axis=0)
        print('Batch numbers:', cnt)
        cnt += 1
    path_save = './embedding/' + params.datetime + '/'
    save_folder = os.path.dirname(path_save)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.savetxt(path_save + 'epoch_' + str(nepoch) + '.txt', embedding_vectors)


if __name__ == '__main__':
    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    # path_file = './data/newres_funcalls_words_jul28.out'
    path_file = './data/small_newres_funcalls_words_jul28.out'
    commits = extract_commit(path_file=path_file)
    commits = reformat_commit_code(commits=commits, num_file=1, num_hunk=8,
                                   num_loc=10, num_leng=120)
    input_option.filter_sizes = [int(k) for k in input_option.filter_sizes.split(',')]
    batches, model = collect_batches(commits=commits, params=input_option)
    for epoch in range(input_option.start_epoch, input_option.end_epoch + 1):
        path_model = './snapshot/' + input_option.datetime + '/epoch_' + str(epoch) + '.pt'
        commit_embedding(path=path_model, batches=batches, model=model, params=input_option, nepoch=epoch)
