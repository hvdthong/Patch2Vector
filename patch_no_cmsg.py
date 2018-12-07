from ultis import extract_commit, write_file, load_file

# print commit id if the commit message has length 0
# check for the original and simplified commit message
if __name__ == '__main__':
    # path_file = './data/newres_funcalls_words_jul28.out'
    # commits = extract_commit(path_file=path_file)
    #
    # id_null_msg = [c['id'] for c in commits if c['msg'] == '']
    # print(len(id_null_msg))
    # # write_file(path_file='./data/id_empty_original_msg.txt', data=id_null_msg)
    # write_file(path_file='./data/id_empty_simplified_msg.txt', data=id_null_msg)

    path_file_org_msg = './data/id_empty_original_msg.txt'
    path_file_simplified_msg = './data/id_empty_simplified_msg.txt'

    id_org, id_sim = load_file(path_file=path_file_org_msg), load_file(path_file=path_file_simplified_msg)
    id_all = list(set(id_sim + id_sim))
    print(len(id_all))
    write_file(path_file='./data/id_empty_all_msg.txt', data=id_all)