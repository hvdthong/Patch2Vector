from main_DCNN import read_args
from Jiang_baseline_data_preparation import load_Jiang_code_data
from ultis import load_file, write_file
from kNN_embedding import make_features, get_data_index
from kNN_embedding import load_kNN_model


def load_code_data(pfile):
    data = load_file(path_file=pfile)
    new_data = list()
    for d in data:
        d_split = d.split('<nl>')
        new_lines = list()
        for l in d_split:
            if l.strip().startswith('-') or l.strip().startswith('+'):
                new_lines.append(l.strip()[1:].strip())
            elif l == '':
                pass
            else:
                new_lines.append(l.strip())
        new_data.append(' '.join(new_lines))
    return new_data


def fold_data(datetime, num_epoch, message, org_code):
    idx_train = [i for i in range(26208)]
    idx_test = [i for i in range(26208, 29208)]
    patch_embedding = make_features(datetime=datetime, num_epoch=num_epoch)
    diff_train, diff_test = patch_embedding[idx_train], patch_embedding[idx_test]
    ref_train, ref_test = get_data_index(data=message, indexes=idx_train), get_data_index(data=message,
                                                                                          indexes=idx_test)
    org_diff_train, org_diff_test = get_data_index(data=org_code, indexes=idx_train), get_data_index(data=org_code,
                                                                                                     indexes=idx_test)
    diff_data, ref_data = (diff_train, diff_test), (ref_train, ref_test)
    org_diff_data = (org_diff_train, org_diff_test)
    return org_diff_data, diff_data, ref_data


if __name__ == '__main__':
    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    # loading the commit code
    ##################################################################################
    ##################################################################################
    path_train_diff = './data/2017_ASE_Jiang/train.26208.diff'
    # data_train_diff = load_Jiang_code_data(pfile=path_train_diff)
    data_train_diff = load_code_data(pfile=path_train_diff)
    path_test_diff = './data/2017_ASE_Jiang/test.3000.diff'
    # data_test_diff = load_Jiang_code_data(pfile=path_test_diff)
    data_test_diff = load_code_data(pfile=path_test_diff)
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
    # datetime, num_epoch = '2018-12-12_01-04-31', 1
    msgs, codes = data_diff, data_msg
    org_diff_data, tf_diff_data, ref_data = fold_data(datetime=datetime, num_epoch=num_epoch, message=msgs,
                                                      org_code=codes)
    k_nearest_neighbor = 5
    blue_scores = load_kNN_model(org_diff_code=org_diff_data, tf_diff_code=tf_diff_data, ref_msg=ref_data,
                                 topK=k_nearest_neighbor, datetime=datetime, num_epoch=num_epoch)

    directory = './blue_scores/' + datetime + '/epoch_' + str(num_epoch) + '.txt'
    write_file(path_file=directory, data=blue_scores)