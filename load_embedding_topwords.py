from ultis import extract_commit, reformat_commit_code, load_file
from main_DCNN import read_args
from padding import padding_commit_topwords, padding_label_topwords
from ultis import mini_batches_topwords, select_commit_based_topwords
import os
import datetime
import torch
from PatchEmbedding_CNN import PatchEmbedding
import numpy as np


def collect_batches(commits, params):
    pad_added_code, pad_removed_code, dict_code = padding_commit_topwords(commits=commits, params=params)
    print('Dictionary of commit code has size: %i' % (len(dict_code)))
    print(pad_added_code.shape, pad_removed_code.shape)

    labels = padding_label_topwords(commits=commits, topwords=topwords)
    batches = mini_batches_topwords(X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=labels,
                                    mini_batch_size=params.batch_size)
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_code = len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    print("\nParameters:")
    for attr, value in sorted(params.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PatchEmbedding(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    return batches, model


def commit_embedding(path, batches, model, params, nepoch):
    model.load_state_dict(torch.load(path))
    embedding_vectors, cnt = list(), 0
    print(path)
    for batch in batches:
        pad_added_code, pad_removed_code, labels = batch
        if torch.cuda.is_available():
            pad_added_code, pad_removed_code, labels = torch.tensor(pad_added_code).cuda(), torch.tensor(
                pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)
        else:
            pad_added_code, pad_removed_code, labels = torch.tensor(pad_added_code).long(), torch.tensor(
                pad_removed_code).long(), torch.tensor(labels).float()

        # predict = model.forward(pad_msg, pad_added_code, pad_removed_code)
        commits_vector = model.forward_commit_embeds(pad_added_code, pad_removed_code)

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

    path_file = './data/newres_funcalls_words_jul28.out'
    # path_file = './data/small_newres_funcalls_words_jul28.out'
    commits = extract_commit(path_file=path_file)
    commits = reformat_commit_code(commits=commits, num_file=1, num_hunk=8,
                                   num_loc=10, num_leng=120)

    ## choose the commits which have the commit message in the top words.
    path_topwords = './data/top_words_commitmsg_1000.txt'
    topwords = load_file(path_file=path_topwords)
    commits = select_commit_based_topwords(words=topwords, commits=commits)

    input_option.filter_sizes = [int(k) for k in input_option.filter_sizes.split(',')]
    batches, model = collect_batches(commits=commits, params=input_option)
    for epoch in range(input_option.start_epoch, input_option.end_epoch + 1):
        path_model = './snapshot/' + input_option.datetime + '/epoch_' + str(epoch) + '.pt'
        commit_embedding(path=path_model, batches=batches, model=model, params=input_option, nepoch=epoch)
