from ultis import extract_commit, reformat_commit_code
from extracting import extract_msg, extract_code


if __name__ == '__main__':
    path_file = './data/newres_funcalls_words_jul28.out'
    commits = extract_commit(path_file=path_file)
    commits = reformat_commit_code(commits=commits, num_file=1, num_hunk=8,
                                   num_loc=10, num_leng=120)

    msgs = extract_msg(commits=commits)
    codes = extract_code(commits=commits)
    print(len(msgs), len(codes))
    exit()
    print(len(commits))
    exit()
    print(type(commits))
    for c in commits:
        print(type(c))
        exit()