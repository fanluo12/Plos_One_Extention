import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import json
import csv

def read_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for json_obj in f:
            data.append(json.loads(json_obj))
    return data


def get_all_tweets(data):
    tweets = []
    for user in data:
        full_tweets = []
        for tweet in user["text"]:
            full_tweets.extend(tweet)
        tweets.append(" ".join(full_tweets))
    return tweets


def write_results(output_path, results):
    with open(os.path.join(output_path, "results.tsv"), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerows(results)

def ApEn(U, m, r):
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)
    return abs(_phi(m + 1) - _phi(m))
  
  def sampen(L, m, r):
    N = len(L)
    B = 0.0
    A = 0.0
    
    # Split time series and save all templates of length m
    xmi = np.array([L[i : i + m] for i in range(N - m)])
    xmj = np.array([L[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([L[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B)
  
  def main(input_file_path, output_file_path):
    # Read the data
    train_data = read_data(os.path.join(input_file_path, "train_tokenized.jsonl"))
    test_data = read_data(os.path.join(input_file_path, "test_tokenized.jsonl"))


    tweets_train = get_all_tweets(train_data)
    has_attempt_train = [data_json["label"] for data_json in train_data]

    count_vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w+', ngram_range=(1, 2))
    bow = dict()
    bow["train"] = (count_vectorizer.fit_transform(tweets_train), has_attempt_train)

    lr_classifier = LogisticRegression(solver='liblinear')
    lr_classifier.fit(*bow["train"])

    output = []
    for user in test_data:
        probs = lr_classifier.predict_proba(count_vectorizer.transform(get_all_tweets([user])))
        output.append([user["id"], lr_classifier.classes_[probs[0].argmax()], probs[0][1]])

    write_results(output_file_path, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenizes the data for the shared task')
    parser.add_argument('--input', help='the directory with the input files', type=str)
    parser.add_argument('--output', help='the directory where the output files should go',
                        type=str)
    args = parser.parse_args()
    main(args.input, args.output)
