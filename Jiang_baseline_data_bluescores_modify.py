from ultis import load_file
import numpy as np


def bluescores_bestscore(datetime, start_epoch, end_epoch):
    print('Datetime: ' + datetime)
    blue_scores_all = list()
    for i in range(start_epoch, end_epoch + 1):
        path_file = './blue_scores/' + datetime + '/epoch_' + str(i) + '.txt'
        blue_scores = load_file(path_file=path_file)
        blue_scores = [float(b) for b in blue_scores]
        blue_scores_all.append(np.array(blue_scores))
    blue_scores_all = np.array(blue_scores_all)
    blue_scores_all = np.reshape(blue_scores_all, (blue_scores_all.shape[1], blue_scores_all.shape[0]))
    max_bluescores = np.amax(blue_scores_all, axis=1)
    print(np.mean(max_bluescores))


if __name__ == '__main__':
    datetime, start_epoch, end_epoch = '2018-12-12_01-04-31', 49, 50
    bluescores_bestscore(datetime=datetime, start_epoch=start_epoch, end_epoch=end_epoch)