from main_DCNN import read_args
from ultis import extract_commit, reformat_commit_code
from tf_idf_words import tfidf_topwords
from textblob import TextBlob as tb
from tf_idf_words import tfidf
import sys
from ultis import write_file


def extract_commit_message(path_file):
    commits = extract_commit(path_file=path_file)
    commits = reformat_commit_code(commits=commits, num_file=1, num_hunk=8,
                                   num_loc=10, num_leng=120)
    return [(c['id'], c['msg']) for c in commits if (len(c['msg'].split(',')) >= 5)]


if __name__ == '__main__':
    start, end = int(sys.argv[1]), int(sys.argv[2])

    path_file = './data/newres_funcalls_words_jul28.out'
    # path_file = './data/small_newres_funcalls_words_jul28.out'
    commit_msg = extract_commit_message(path_file=path_file)
    top_words = 3

    docs = [tb(c[1].replace(',', ' ').strip()) for c in commit_msg]  # index 0 is an ID , index 1 is a message
    data_print = list()
    for i in range(start, end):
        print("Top words in document {}".format(i))
        data_print.append('Top words in document {}'.format(i))
        scores = {word: tfidf(word, docs[i], docs) for word in docs[i].words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:top_words]:
            print("Word: {}, TF-IDF: {}".format(word, round(score, 5)))
            data_print.append("Word: {}, TF-IDF: {}".format(word, round(score, 5)))

    write_file(path_file='./data/top_words_commitmsg_top' + str(top_words) + '_' + str(start) + '_' + str(end),
               data=data_print)
    print('./data/top_words_commitmsg_top' + str(top_words) + '_' + str(start) + '_' + str(end))
