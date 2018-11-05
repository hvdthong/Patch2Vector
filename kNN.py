from ultis import extract_commit, reformat_commit_code
from extracting import extract_msg, extract_code
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer


def empty_info(message, code):
    return [i for i in range(0, len(message)) if (message[i] == '' or code[i] == '')]


def del_info(data, indexes):
    for i in sorted(indexes, reverse=True):
        del data[i]
    return data


def reformat_data(data):
    msgs = extract_msg(commits=data)
    codes = extract_code(commits=data)
    indexes = empty_info(message=msgs, code=codes)
    msgs, codes = del_info(data=msgs, indexes=indexes), del_info(data=codes, indexes=indexes)
    return msgs, codes


def run_kfold(data, fold):
    kf = KFold(n_splits=fold)
    for train, test in kf.split(data):
        return train, test


def get_data_index(data, indexes):
    return [data[i] for i in indexes]


def fold_data(message, code, fold):
    idx_train, idx_test = run_kfold(data=message, fold=fold)
    diff_train, diff_test = code[idx_train], code[idx_test]
    ref_train, ref_test = get_data_index(data=message, indexes=idx_train), get_data_index(data=message,
                                                                                          indexes=idx_test)
    diff_data, ref_data = (diff_train, diff_test), (ref_train, ref_test)
    return diff_data, ref_data


def make_features(data):
    vectorizer = CountVectorizer()
    data = vectorizer.fit_transform(data)
    return data


if __name__ == '__main__':
    path_file = './data/newres_funcalls_words_jul28.out'
    commits = extract_commit(path_file=path_file)
    commits = reformat_commit_code(commits=commits, num_file=1, num_hunk=8,
                                   num_loc=10, num_leng=120)
    msgs, codes = reformat_data(data=commits)
    nfold = 5  # number of cross-validation
    fold_data(message=msgs, code=make_features(data=codes), fold=nfold)
