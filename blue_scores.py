from ultis import load_file
import statistics


def blue_scores_embedding(datetime, start_epoch, end_epoch):
    print('Datetime: ' + datetime)
    for i in range(start_epoch, end_epoch + 1):
        path_file = './blue_scores/' + datetime + '/epoch_' + str(i) + '.txt'
        blue_scores = load_file(path_file=path_file)
        blue_scores = [float(b) for b in blue_scores]
        print('Epoch: ' + str(i) + ' Mean of blue scores: ' + str(statistics.mean(blue_scores)))
        # print('Epoch: ' + str(i) + ' Std of blue scores: ' + str(statistics.stdev(blue_scores)))


if __name__ == '__main__':
    path_blue_score = './knn_blue_scores_test_file.txt'
    blue_scores = load_file(path_file=path_blue_score)
    blue_scores = [float(b) for b in blue_scores]
    print('Mean of blue scores: ' + str(statistics.mean(blue_scores)))
    # print('Std of blue scores: ' + str(statistics.stdev(blue_scores)))

    datetime, start_epoch, end_epoch = '2018-11-30_06-28-52', 1, 50
    blue_scores_embedding(datetime=datetime, start_epoch=start_epoch, end_epoch=end_epoch)

    datetime, start_epoch, end_epoch = '2018-11-30_06-29-51', 1, 50
    blue_scores_embedding(datetime=datetime, start_epoch=start_epoch, end_epoch=end_epoch)