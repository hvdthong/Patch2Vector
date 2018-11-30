from padding import padding_commit_topwords, padding_label_topwords
import os
import datetime
from ultis import mini_batches_topwords
import torch
from PatchEmbedding_CNN import PatchEmbedding
import torch.nn as nn


def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}.pt'.format(save_prefix, epochs)
    torch.save(model.state_dict(), save_path)


def running_train(batches, model, params):
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    steps, num_epoch = 0, 1
    for epoch in range(1, params.num_epochs + 1):
        for batch in batches:
            pad_added_code, pad_removed_code, labels = batch
            if torch.cuda.is_available():
                pad_added_code, pad_removed_code, labels = torch.tensor(pad_added_code).cuda(), torch.tensor(
                    pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)
            else:
                pad_added_code, pad_removed_code, labels = torch.tensor(pad_added_code).long(), torch.tensor(
                    pad_removed_code).long(), torch.tensor(labels).float()

            optimizer.zero_grad()
            predict = model.forward(pad_added_code, pad_removed_code)
            loss = nn.BCELoss()
            loss = loss(predict, labels)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % params.log_interval == 0:
                print('\rEpoch: {} step: {} - loss: {:.6f}'.format(num_epoch, steps, loss.item()))

        save(model, params.save_dir, 'epoch', num_epoch)
        num_epoch += 1


def train_model_topwords(commits, params, topwords):
    pad_added_code, pad_removed_code, dict_code = padding_commit_topwords(commits=commits, params=params)
    print('Dictionary of commit code has size: %i' % (len(dict_code)))
    print(pad_added_code.shape, pad_removed_code.shape)

    labels = padding_label_topwords(commits=commits, topwords=topwords)
    batches = mini_batches_topwords(X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=labels,
                                    mini_batch_size=params.batch_size)

    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    params.vocab_code = len(dict_code)
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
