from ultis import write_file

if __name__ == '__main__':
    datetime = '2018-11-30_06-28-52'
    embedding_dim = 32
    filter_sizes = '1,2'
    hidden_units = 128
    num_filters = 32
    start_epoch, end_epoch = 1, 50

    data = list()
    for i in range(start_epoch, end_epoch + 1):
        print('python kNN_embedding.py -datetime ' + datetime + ' -embedding_dime ' + str(
            embedding_dim) + ' -filter_sizes ' + filter_sizes + ' -hidden_units ' + str(
            hidden_units) + ' -num_filters ' + str(num_filters) + ' -start_epoch ' + str(i))
        data.append('python kNN_embedding.py -datetime ' + datetime + ' -embedding_dime ' + str(
            embedding_dim) + ' -filter_sizes ' + filter_sizes + ' -hidden_units ' + str(
            hidden_units) + ' -num_filters ' + str(num_filters) + ' -start_epoch ' + str(i))
    write_file(path_file=datetime + '.txt', data=data)