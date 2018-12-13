from main_DCNN import read_args
from Jiang_baseline_data_preparation import load_Jiang_code_data
from ultis import load_file


def load_code_data(pfile):
    data = load_file(path_file=pfile)
    new_data = list()
    for d in data:
        d_split = d.split('<nl>')
        new_lines = list()
        for l in d_split:
            if l.strip().startswith('-') or l.strip().startswith('+'):
                new_lines.append(l.strip()[1:])
            else:
                new_lines.append(l.strip())
        new_data.append(new_lines)
    return new_data


if __name__ == '__main__':
    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    # loading the commit code
    ##################################################################################
    ##################################################################################
    path_train_diff = './data/2017_ASE_Jiang/train.26208.diff'
    data_train_diff = load_Jiang_code_data(pfile=path_train_diff)
    data_train_diff = load_code_data(pfile=path_train_diff)
    path_test_diff = './data/2017_ASE_Jiang/test.3000.diff'
    data_test_diff = load_Jiang_code_data(pfile=path_test_diff)
    data_diff = data_train_diff + data_test_diff
    padding_code_info = (15, 40)  # max_line = 15; max_length = 40
    ##################################################################################
    ##################################################################################

    # loading the commit msg
    ##################################################################################
    ##################################################################################
    path_train_msg = './data/2017_ASE_Jiang/train.26208.msg'
    data_train_msg = load_file(path_file=path_train_msg)
    path_test_msg = './data/2017_ASE_Jiang/test.3000.msg'
    data_test_msg = load_file(path_file=path_test_msg)
    data_msg = data_train_msg + data_test_msg

    datetime, num_epoch = input_option.datetime, input_option.start_epoch
