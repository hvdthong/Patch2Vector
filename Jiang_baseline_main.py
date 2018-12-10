from ultis import load_file

if __name__ == '__main__':
    path_train = './data/2017_ASE_Jiang/train.26208.diff'
    path_test = './data/2017_ASE_Jiang/test.3000.diff'
    train, test = load_file(path_file=path_train), load_file(path_file=path_test)
    data = train + test
    print(len(train), len(test), len(data))