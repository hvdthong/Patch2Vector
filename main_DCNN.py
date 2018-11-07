from ultis import extract_commit, reformat_commit_code
from train import train_model

if __name__ == '__main__':
    path_file = './data/newres_funcalls_words_jul28.out'
    commits = extract_commit(path_file=path_file)
    commits = reformat_commit_code(commits=commits, num_file=1, num_hunk=8,
                                   num_loc=10, num_leng=120)

    print(len(commits))