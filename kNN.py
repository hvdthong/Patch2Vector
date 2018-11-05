from ultis import extract_commit, reformat_commit_code
from extracting import extract_msg, extract_code


def empty_info(message, code):
    return [i for i in range(0, len(message)) if (message[i] == '' or code[i] == '')]


def del_info(data, indexes):
    for i in indexes:
        del data[i]
    return data


if __name__ == '__main__':
    path_file = './data/newres_funcalls_words_jul28.out'
    commits = extract_commit(path_file=path_file)
    commits = reformat_commit_code(commits=commits, num_file=1, num_hunk=8,
                                   num_loc=10, num_leng=120)

    msgs = extract_msg(commits=commits)
    codes = extract_code(commits=commits)
    print(len(empty_info(message=msgs, code=codes)))
    print(empty_info(message=msgs, code=codes))
    print(len(msgs), len(codes))
    exit()
    print(len(commits))
    exit()
    print(type(commits))
    for c in commits:
        print(type(c))
        exit()
