from main_DCNN import read_args
from ultis import extract_commit, reformat_commit_code, load_file, select_commit_based_topwords
from train_topwords import train_model_topwords


if __name__ == '__main__':
    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    path_file = './data/newres_funcalls_words_jul28.out'
    # path_file = './data/small_newres_funcalls_words_jul28.out'
    commits = extract_commit(path_file=path_file)
    commits = reformat_commit_code(commits=commits, num_file=1, num_hunk=8,
                                   num_loc=10, num_leng=120)

    ## choose the commits which have the commit message in the top words.
    # path_topwords = './data/top_words_commitmsg_1000.txt'
    path_topwords = './data/top_words_commitmsg_all.txt'
    # path_topwords = './data/top_words_commitmsg_simplified.txt'
    topwords = load_file(path_file=path_topwords)
    commits = select_commit_based_topwords(words=topwords, commits=commits)

    train_model_topwords(commits=commits, params=input_option, topwords=topwords)