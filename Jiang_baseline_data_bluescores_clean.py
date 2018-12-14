from ultis import load_file
import statistics
from Jiang_baseline_data_message_pattern import matching_regular_expression
import numpy as np


def bluescore_clean(indexes, datetime, start_epoch, end_epoch):
    print('Datetime: ' + datetime)
    for i in range(start_epoch, end_epoch + 1):
        path_file = './blue_scores/' + datetime + '/epoch_' + str(i) + '.txt'
        blue_scores = load_file(path_file=path_file)
        blue_scores = [float(b) for b in blue_scores]
        blue_scores = [blue_scores[i] for i in indexes]
        print('Epoch: ' + str(i) + ' Mean of blue scores: ' + str(statistics.mean(blue_scores)))


def bluescore_clean_modify(indexes, datetime, start_epoch, end_epoch):
    print('Datetime: ' + datetime)
    bluescores_all = list()
    for i in range(start_epoch, end_epoch + 1):
        path_file = './blue_scores/' + datetime + '/epoch_' + str(i) + '.txt'
        blue_scores = load_file(path_file=path_file)
        blue_scores = [float(b) for b in blue_scores]
        blue_scores = [blue_scores[i] for i in indexes]
        bluescores_all.append(np.array(blue_scores))
    blue_scores_all = np.array(bluescores_all)
    blue_scores_all = np.reshape(blue_scores_all, (blue_scores_all.shape[1], blue_scores_all.shape[0]))
    max_bluescores = np.amax(blue_scores_all, axis=1)
    print('Mean of blue scores: %f' % (np.mean(max_bluescores)))


if __name__ == '__main__':
    path_test_msg = './data/2017_ASE_Jiang/test.3000.msg'
    data_test_msg = load_file(path_file=path_test_msg)
    data_test_msg = matching_regular_expression(data=data_test_msg)
    print('Testing data size: %i' % len(data_test_msg))

    # datetime, start_epoch, end_epoch = '2018-12-12_01-04-31', 1, 50
    # bluescore_clean(indexes=data_test_msg, datetime=datetime, start_epoch=start_epoch, end_epoch=end_epoch)

    datetime, start_epoch, end_epoch = '2018-12-12_01-04-31', 49, 50
    bluescore_clean_modify(indexes=data_test_msg, datetime=datetime, start_epoch=start_epoch, end_epoch=end_epoch)
