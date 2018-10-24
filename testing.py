def load_file(path):
    with open(path) as f:
        content = f.readlines()
    return [x.strip() for x in content]


if __name__ == '__main__':
    print('hello')
    print('testing vim')

    path_ = './data/newres_jul28.dict'
    data = load_file(path=path_)
    print(len(data))

    path_ = './data/newres_funcalls_jul28.out'
    data = load_file(path=path_)
    print(len(data))
