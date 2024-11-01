#
#
#
#

from collections import Counter
from statistics import mean
import csv


def read_corpus(corpus_file):
    '''Reads the given corpus, returns the documents and labels (OFF or NOT) in a list.'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            documents.append(tokens[:-1])
            # 2-class problem: OFF vs NOT
            labels.append(tokens[-1])
    return documents, labels


def main():
    x_doc, x_lab = read_corpus('data/long_test.tsv')

    short_tweets = []
    long_tweets = []

    for i in range(len(x_doc)):
        twt = " ".join(x_doc[i])
        twt_len = len(twt)

        if twt_len <= 80 and len(short_tweets) < 220:
            short_tweets.append(twt + ' ' + x_lab[i])
        
        if twt_len >= 140 and len(long_tweets) < 220:
            long_tweets.append(twt + ' ' + x_lab[i])
    
    # print(short_tweets)

    # with open('data/short_test.tsv', 'w') as f_output:
    #     for tweet in short_tweets:
    #        f_output.write(tweet + '\n')
    
    # with open('data/long_test.tsv', 'w') as f_output:
    #     for tweet in long_tweets:
    #         f_output.write(tweet + '\n')

    print(Counter(x_lab))


if __name__ == "__main__":
    main()