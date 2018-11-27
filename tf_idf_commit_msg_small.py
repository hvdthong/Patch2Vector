from os import listdir
from os.path import isfile, join
from ultis import load_file, write_file
import collections


def document_index(data):
    doc_index = list()
    for i in range(0, len(data)):
        if data[i].startswith('Top'):
            doc_index.append(i)
    return doc_index


def get_top_words(data):
    return [data[i].split()[1].replace(',', '').strip() for i in range(1, len(data))]


def build_dict_topwords(data, top_K):
    top_words = list()
    doc_indexes = document_index(data=data)
    for i in range(0, len(doc_indexes)):
        if i == len(doc_indexes) - 1:
            top_words += get_top_words(data=data[doc_indexes[i]:])
        else:
            top_words += get_top_words(data=data[doc_indexes[i]:doc_indexes[i + 1]])
    counter = collections.Counter(top_words)
    top_words_k = [word[0] for word in counter.most_common(top_K)]
    write_file(path_file='./data/top_words_commitmsg_' + str(top_K) + '.txt', data=top_words_k)


def load_top_words(paths, top_K):
    data = [load_file(path_file=p) for p in paths]
    data = sum(data, [])
    build_dict_topwords(data=data, top_K=top_K)
    return data


if __name__ == '__main__':
    path_file = './data/top_words_commitmsg'
    files = [f for f in listdir(path_file) if isfile(join(path_file, f))]
    files = [path_file + '/' + f for f in files]
    top_K = 1000
    load_top_words(paths=files, top_K=top_K)
