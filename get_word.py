from ultis import load_file, commits_index


def load_dict(data):
    dict_ = {}
    for d in data:
        d_split = d.split(':')
        dict_[d_split[0]] = d_split[1].strip()
    return dict_


def mapping_commit_msg(msg, dictionary):
    if len(msg) == 0:
        return ''
    else:
        words = msg.split(',')
        new_words = ''
        for w in words:
            new_words += dictionary[w] + ','
        return new_words[:-1]


def mapping_commit_code(code, dictionary):
    split_code = code.split(':')
    new_code = split_code[0] + ':' + split_code[1] + ':' + split_code[2] + ': '
    for c in split_code[3].strip().split(','):
        new_code += dictionary[c] + ','
    return new_code[:-1]


def mapping_word_commit(commit, dictionary):
    new_line = list()
    flag_code = False
    for i in range(len(commit)):
        if flag_code is True:
            if len(commit[i]) == 0:
                new_line.append(commit[i])
            elif 'file:' not in commit[i]:
                new_line.append(mapping_commit_code(code=commit[i], dictionary=dictionary))
            else:
                new_line.append(commit[i])
        else:
            if i == 6 or i == 9:
                new_line.append(mapping_commit_msg(msg=commit[i], dictionary=dictionary))
            else:
                if 'commit code:' in commit[i]:
                    new_line.append(commit[i])
                    flag_code = True
                else:
                    new_line.append(commit[i])
    return new_line


def word_commit(path, dictionary):
    commits = load_file(path_file=path)
    indexes = commits_index(commits=commits)
    new_commits = list()
    for i in range(0, len(indexes)):
        if i == len(indexes) - 1:
            commit = mapping_word_commit(dictionary=dictionary, commit=commits[indexes[i]:])
        else:
            commit = mapping_word_commit(dictionary=dictionary, commit=commits[indexes[i]:indexes[i + 1]])
        new_commits += commit
        for c in commit:
            print(c)
    return new_commits


if __name__ == '__main__':
    path_dict = './data/newres_jul28.dict'
    dict_ = load_dict(load_file(path_file=path_dict))

    path_commit = './data/newres_funcalls_jul28.out'
    commits_ = word_commit(path=path_commit, dictionary=dict_)
    # for c in commits_:
    #     print(c)
