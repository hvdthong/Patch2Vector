from ultis import load_file


def get_diff_data(commit, type_code):
    lines = list()
    for c in commit:



def load_Jiang_code_data(pfile):
    data = load_file(path_file=pfile)
    data = [d.split('<nl>') for d in data]
    return data


if __name__ == '__main__':
    path_train_diff = './data/2017_ASE_Jiang/train.26208.diff'
    load_Jiang_code_data(pfile=path_train_diff)

    exit()
    path_test_diff = './data/2017_ASE_Jiang/test.3000.diff'
    train_diff, test_diff = load_file(path_file=path_train_diff), load_file(path_file=path_test_diff)
    data_diff = train_diff + test_diff
    print(len(train_diff), len(test_diff), len(data_diff))

    path_train_msg = './data/2017_ASE_Jiang/train.26208.msg'
    path_test_msg = './data/2017_ASE_Jiang/test.3000.msg'
    train_msg, test_msg = load_file(path_file=path_train_msg), load_file(path_file=path_test_msg)
    data_msg = train_msg + test_msg
    print(len(train_msg), len(test_msg), len(data_msg))
