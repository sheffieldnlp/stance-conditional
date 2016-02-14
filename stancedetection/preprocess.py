__author__ = 'Isabelle Augenstein'

from nltk.corpus import stopwords
import collections
from readwrite import reader
from twokenize_wrapper import tokenize
import numpy as np


def filterStopwords(tokenised_tweet):
    """
    Remove stopwords from tokenised tweet
    :param tokenised_tweet: tokenised tweet
    :return: tweet tokens without stopwords
    """
    stops = stopwords.words("english")
    # extended with string.punctuation and rt and #semst, removing links further down
    stops.extend(["!", "\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":",
                  ";", "<", "=", ">", "?", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~"])
    stops.extend(["rt", "#semst", "thats", "im", "'s", "...", "via"])
    stops = set(stops)
    return [w for w in tokenised_tweet if (not w in stops and not w.startswith("http"))]


def build_dataset(words, vocabulary_size=50000):
    """
    Build vocabulary, code based on tensorflow/examples/tutorials/word2vec/word2vec_basic.py
    :param words: list of words in corpus
    :param vocabulary_size: max vocabulary size
    :return: counts, dictionary mapping words to indeces, reverse dictionary
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return count, dictionary, reverse_dictionary


def transform_tweet_nopadding(dictionary, words):
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    return data


def transform_tweet(dictionary, words, maxlen=20):
    """
    Transform list of tokens, add padding to maxlen
    :param dictionary: dict which maps tokens to integer indices
    :param words: list of tokens
    :param maxlen: maximum length
    :return: transformed tweet, as numpy array
    """
    data = list()
    for i in range(0, maxlen-1):
        if i < len(words):
            word = words[i]
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
        else:
            index = 0
        data.append(index)
    return np.asarray(data)


def tokenise_tweets(tweets):
    return [filterStopwords(tokenize(tweet.lower())) for tweet in tweets]


def transform_labels(labels, dim=3):
    labels_t = []
    for lab in labels:
        v = np.zeros(dim)
        if lab == 'NONE':
            ix = 0
        elif lab == 'AGAINST':
            ix = 1
        elif lab == 'FAVOR':
            ix = 2
        v[ix] = 1
        labels_t.append(v)
    return labels_t


if __name__ == '__main__':
    tweets, targets, labels = reader.readTweetsOfficial("../data/semeval2016-task6-train+dev.txt")
    tokens = tokenise_tweets(tweets)
    count, dictionary, reverse_dictionary = build_dataset([token for senttoks in tokens for token in senttoks])  #flatten tweets for vocab construction
    transformed_tweets = [transform_tweet(dictionary, senttoks) for senttoks in tokens]
    transformed_labels = transform_labels(labels)
    print('Longest tweet', len(max(transformed_tweets,key=len)))
    print('Most common words (+UNK)', count[:5])
    #print('Sample data', data[:10])