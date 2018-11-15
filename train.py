#  * This file is part of PatchNet, licensed under the terms of the GPL v2.
#  * See copyright.txt in the PatchNet source code for more information.
#  * The PatchNet source code can be obtained at
#  * https://github.com/hvdthong/PatchNetTool

from padding import padding_commit
import os
import datetime
from ultis import mini_batches
import torch
from PatchNet_CNN import PatchNet
import torch.nn as nn
from sklearn.metrics import accuracy_score


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
            pad_msg, pad_added_code, pad_removed_code, labels = batch
            if torch.cuda.is_available():
                pad_msg, pad_added_code, pad_removed_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_added_code).cuda(), torch.tensor(pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)
            else:
                pad_msg, pad_added_code, pad_removed_code, labels = torch.tensor(pad_msg).long(), torch.tensor(
                    pad_added_code).long(), torch.tensor(pad_removed_code).long(), torch.tensor(labels).float()

            optimizer.zero_grad()
            predict = model.forward(pad_msg, pad_added_code, pad_removed_code)
            loss = nn.BCELoss()
            loss = loss(predict, labels)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % params.log_interval == 0:
                print('\rEpoch: {} step: {} - loss: {:.6f}'.format(num_epoch, steps, loss.item()))

            if steps % params.test_interval == 0:
                if torch.cuda.is_available():
                    predict, labels = predict.cpu().detach().numpy(), labels.cpu().detach().numpy()
                else:
                    predict, labels = predict.detach().numpy(), labels.detach().numpy()
                predict = [1 if p >= 0.5 else 0 for p in predict]
                accuracy = accuracy_score(y_true=labels, y_pred=predict)
                print(
                    '\rEpoch: {} Step: {} - loss: {:.6f}  acc: {:.4f}'.format(num_epoch, steps, loss.item(), accuracy))

        save(model, params.save_dir, 'epoch', num_epoch)
        num_epoch += 1


def train_model(commits, params):
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
    if torch.cuda.is_available():
        model = model.cuda()

    running_train(batches=batches, model=model, params=params)
