from main_DCNN import read_args
from ultis import extract_commit, reformat_commit_code
from tf_idf_words import tfidf_topwords


def extract_commit_message(path_file):
    commits = extract_commit(path_file=path_file)
    commits = reformat_commit_code(commits=commits, num_file=1, num_hunk=8,
                                   num_loc=10, num_leng=120)
    return [(c['id'], c['msg']) for c in commits if (len(c['msg'].split(',')) >= 5)]


if __name__ == '__main__':
    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    path_file = './data/newres_funcalls_words_jul28.out'
    # path_file = './data/small_newres_funcalls_words_jul28.out'
    commit_msg = extract_commit_message(path_file=path_file)
    top_words = 3
    tfidf_topwords(commits=commit_msg, top_words=top_words)
