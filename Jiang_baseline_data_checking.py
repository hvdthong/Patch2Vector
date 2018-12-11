from Jiang_baseline_data_preparation import load_Jiang_code_data
import matplotlib.pyplot as plt
import statistics
from ultis import load_file


def checking_number_lines(data):
    nlines_removed = [len(d['removed']) for d in data]
    nlines_added = [len(d['added']) for d in data]
    nlines = nlines_removed + nlines_added
    print('Mean of number of lines: %f' % (statistics.mean(nlines)))
    print('Varience of number of lines: %f' % (statistics.variance(nlines)))
    print('Max length: %f' % (max(nlines)))
    plt.hist(nlines, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()


def checking_length_of_line(data):
    nlength_lines_removed = [len(l.split()) for d in data for l in d['removed']]
    nlength_lines_added = [len(l.split()) for d in data for l in d['added']]
    nlength_lines = nlength_lines_removed + nlength_lines_added
    print('Mean of length: %f' % (statistics.mean(nlength_lines)))
    print('Varience of length: %f' % (statistics.variance(nlength_lines)))
    print('Max length: %f' % (max(nlength_lines)))
    plt.hist(nlength_lines, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()


def checking_commit_msg(data):
    ntokens = [len(d.split()) for d in data]
    print('Mean of length: %f' % (statistics.mean(ntokens)))
    print('Varience of length: %f' % (statistics.variance(ntokens)))
    print('Max length: %f' % (max(ntokens)))
    plt.hist(ntokens, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()


if __name__ == '__main__':
    # path_train_diff = './data/2017_ASE_Jiang/train.26208.diff'
    # data_train_diff = load_Jiang_code_data(pfile=path_train_diff)
    # path_test_diff = './data/2017_ASE_Jiang/test.3000.diff'
    # data_test_diff = load_Jiang_code_data(pfile=path_test_diff)
    # data_diff = data_train_diff + data_test_diff
    # print(len(data_diff))
    # checking_number_lines(data=data_diff)  # number of lines 15
    # # checking_length_of_line(data=data_diff)  # number of length 40

    path_train_msg = './data/2017_ASE_Jiang/train.26208.msg'
    data_train_msg = load_file(path_file=path_train_msg)
    path_test_msg = './data/2017_ASE_Jiang/test.3000.msg'
    data_test_msg = load_file(path_file=path_test_msg)
    data_msg = data_train_msg + data_test_msg
    checking_commit_msg(data=data_msg)
    print(len(data_msg))
