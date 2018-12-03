from ultis import extract_commit, reformat_commit_code, write_file
import itertools

if __name__ == '__main__':
    # path_file = './data/newres_funcalls_words_jul28.out'
    path_file = './data/small_newres_funcalls_words_jul28.out'
    commits = extract_commit(path_file=path_file)
    commits = reformat_commit_code(commits=commits, num_file=1, num_hunk=8,
                                   num_loc=10, num_leng=120)
    commits_msg = [c['msg'].split(',') for c in commits if len(c['msg']) > 0]

    dict_words = sorted(list(set(list(itertools.chain(*commits_msg)))))
    print(len(dict_words))
    # write_file(path_file='./data/top_words_commitmsg_all.txt', data=dict_words)
    write_file(path_file='./data/top_words_commitmsg_simplified.txt', data=dict_words)