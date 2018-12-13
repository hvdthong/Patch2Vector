import re
from ultis import load_file


def list_regular_expression(line):
    match_1 = bool(re.match(r'^ignore update \' .* \.$', line))
    match_2 = bool(re.match(r'^update(d)? (changelog|gitignore|readme( . md| file)?)( \.)?$', line))
    match_3 = bool(re.match(r'^prepare version (v)?[ \d.]+$', line))
    match_4 = bool(re.match(r'^bump (up )?version( number| code)?( to (v)?[ \d.]+( - snapshot)?)?( \.)?$', line))
    match_5 = bool(re.match(r'^modify (dockerfile|makefile)( \.)?$', line))
    match_6 = bool(re.match(r'^update submodule(s)?( \.)?$', line))
    return match_1, match_2, match_3, match_4, match_5, match_6


def matching_regular_expression(data):
    new_data = list()
    for i in range(len(data)):
        d = data[i]
        match_1, match_2, match_3, match_4, match_5, match_6 = list_regular_expression(line=d.lower().strip())
        if (match_1 is False) and (match_2 is False) and (match_3 is False) and (match_4 is False) and (
                match_5 is False) and (match_6 is False):
            new_data.append(i)
    return new_data


if __name__ == '__main__':
    path_train_msg = './data/2017_ASE_Jiang/train.26208.msg'
    data_train_msg = load_file(path_file=path_train_msg)
    path_test_msg = './data/2017_ASE_Jiang/test.3000.msg'
    data_test_msg = load_file(path_file=path_test_msg)
    data_msg = data_train_msg + data_test_msg

    print('Training data size: %i ' % len(data_train_msg))
    print('Testing data size: %i' % len(data_test_msg))

    data_train_msg = matching_regular_expression(data=data_train_msg)
    data_test_msg = matching_regular_expression(data=data_test_msg)
    print('Training data size: %i ' % len(data_train_msg))
    print('Testing data size: %i' % len(data_test_msg))