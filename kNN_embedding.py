from ultis import extract_commit, reformat_commit_code, load_file, select_commit_based_topwords, write_file
from extracting import extract_msg, extract_code
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import os
import statistics
from main_DCNN import read_args
import nltk


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


def fold_data(datetime, num_epoch, message, org_code, fold):
    idx_train, idx_test = run_kfold(data=message, fold=fold)
    patch_embedding = make_features(datetime=datetime, num_epoch=num_epoch)
    diff_train, diff_test = patch_embedding[idx_train], patch_embedding[idx_test]
    ref_train, ref_test = get_data_index(data=message, indexes=idx_train), get_data_index(data=message,
                                                                                          indexes=idx_test)
    org_diff_train, org_diff_test = get_data_index(data=org_code, indexes=idx_train), get_data_index(data=org_code,
                                                                                                     indexes=idx_test)
    diff_data, ref_data = (diff_train, diff_test), (ref_train, ref_test)
    org_diff_data = (org_diff_train, org_diff_test)
    return org_diff_data, diff_data, ref_data


def make_features(datetime, num_epoch):
    embedding = np.loadtxt('./embedding/' + datetime + '/epoch_' + str(num_epoch) + '.txt')
    return embedding


def cosine_similarity_score(train_data, element, index, datetime, num_epoch):
    element = np.reshape(element, (1, element.shape[0]))

    sim_score = cosine_similarity(X=train_data, Y=element)
    directory = './cosine_embedding/' + datetime + '/epoch_' + str(num_epoch)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savetxt(directory + '/test_' + str(index) + '.txt', sim_score)
    print(index)


def kNN_model(tf_diff_code, datetime, num_epoch):
    tf_diff_train, tf_diff_test = tf_diff_code
    [cosine_similarity_score(train_data=tf_diff_train, element=tf_diff_test[i, :], index=i, datetime=datetime,
                             num_epoch=num_epoch) for i in range(0, tf_diff_test.shape[0])]


def finding_topK(cosine_sim, topK):
    cosine_sim = list(cosine_sim)
    topK_index = list()
    for i in range(topK):
        max_ = max(cosine_sim)
        index = cosine_sim.index(max_)
        topK_index.append(index)
        del cosine_sim[index]
    return topK_index


def finding_bestK(diff_trains, diff_test, topK_index):
    diff_code_train = get_data_index(data=diff_trains, indexes=topK_index)
    diff_code_train = [d.split() for d in diff_code_train]
    diff_code_test = diff_test.split()
    chencherry = SmoothingFunction()
    scores = [sentence_bleu(references=[d], hypothesis=diff_code_test, smoothing_function=chencherry.method1) for d in
              diff_code_train]
    bestK = topK_index[scores.index(max(scores))]
    return bestK


def remove_stopword_punctuation(line):
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens]
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    from nltk.corpus import stopwords
    stopwords = set(stopwords.words('english'))
    words = [w for w in words if not w in stopwords]
    return ' '.join(words).strip()


def load_kNN_model(org_diff_code, tf_diff_code, ref_msg, topK, datetime, num_epoch):
    org_diff_train, org_diff_test = org_diff_code
    tf_diff_train, tf_diff_test = tf_diff_code
    ref_train, ref_test = ref_msg
    blue_scores = list()
    for i in range(tf_diff_test.shape[0]):
        element = tf_diff_test[i, :]
        element = np.reshape(element, (1, element.shape[0]))
        cosine_sim = cosine_similarity(X=tf_diff_train, Y=element)

        # cosine_sim = np.loadtxt(
        #     './cosine_embedding/' + datetime + '/epoch_' + str(num_epoch) + '/test_' + str(i) + '.txt')

        topK_index = finding_topK(cosine_sim=cosine_sim, topK=topK)
        bestK = finding_bestK(diff_trains=org_diff_train, diff_test=org_diff_test[i], topK_index=topK_index)
        train_msg, test_msg = ref_train[bestK].lower(), ref_test[i].lower()
        # train_msg, test_msg = remove_stopword_punctuation(line=train_msg), remove_stopword_punctuation(line=test_msg)
        if (len(train_msg.split()) >= 4) and (len(test_msg.split()) >= 4):
            # blue_score = sentence_bleu(references=[train_msg.split()], hypothesis=test_msg.split())
            chencherry = SmoothingFunction()
            blue_score = sentence_bleu(references=[train_msg.split()], hypothesis=test_msg.split(),
                                       smoothing_function=chencherry.method5)
        else:
            blue_score = sentence_bleu(references=[train_msg.split()], hypothesis=test_msg.split(),
                                       weights=[0.25])
        print('Epoch:%i, index:%i, blue_score:%f ' % (num_epoch, i, blue_score))
        blue_scores.append(blue_score)
    print('Datetime: ' + datetime + ' Epoch: ' + str(num_epoch) + ' Mean of blue scores: ' + str(
        statistics.mean(blue_scores)))
    print('Datetime: ' + datetime + ' Epoch: ' + str(num_epoch) + ' Std of blue scores: ' + str(
        statistics.stdev(blue_scores)))
    return blue_scores


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
    # path_topwords = './data/top_words_commitmsg_all.txt'
    path_topwords = './data/top_words_commitmsg_simplified.txt'
    topwords = load_file(path_file=path_topwords)
    commits = select_commit_based_topwords(words=topwords, commits=commits)

    datetime, num_epoch = input_option.datetime, input_option.start_epoch
    # datetime, num_epoch = '2018-11-30_12-51-04', 1

    msgs, codes = reformat_data(data=commits)
    nfold = 9  # number of cross-validation
    org_diff_data, tf_diff_data, ref_data = fold_data(datetime=datetime, num_epoch=num_epoch, message=msgs,
                                                      org_code=codes, fold=nfold)
    k_nearest_neighbor = 5

    # calculating the cosine similarity between each element in testing data to training data
    # ---------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------
    # kNN_model(tf_diff_code=tf_diff_data, datetime=datetime, num_epoch=num_epoch)

    # loading cosine similarity data and get the top K nearest neighbors, and calculate BLEU-score between diff-code
    # Then we identify the nearest neighbor and get the reference message, we then calculate the BLEU-score between
    # training data and test data based on the reference messages
    # ---------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------
    blue_scores = load_kNN_model(org_diff_code=org_diff_data, tf_diff_code=tf_diff_data, ref_msg=ref_data,
                                 topK=k_nearest_neighbor, datetime=datetime, num_epoch=num_epoch)

    directory = './blue_scores/' + datetime + '/epoch_' + str(num_epoch) + '.txt'
    write_file(path_file=directory, data=blue_scores)
