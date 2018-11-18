from ultis import load_file

if __name__ == '__main__':
    path_blue_score = './knn_blue_scores_test_file.txt'
    blue_scores = load_file(path_file=path_blue_score)
    blue_scores = [float(b) for b in blue_scores]
    print(sum(blue_scores) / len(blue_scores))