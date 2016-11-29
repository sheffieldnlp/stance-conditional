from nltk.corpus import stopwords
import collections
from readwrite import reader
from twokenize_wrapper.twokenize import tokenize
import numpy as np

KEYWORDS = {'clinton': ['hillary', 'clinton'],
            'trump': ['donald trump', 'trump', 'donald'],
            'climate': ['climate'],
            'feminism': ['feminism', 'feminist'],
            'abortion': ['abortion', 'aborting'],
            'atheism': ['atheism', 'atheist']
            }

TOPICS_LONG = {'clinton': 'Hillary Clinton',
               'trump': 'Donald Trump',
               'climate': 'Climate Change is a Real Concern',
               'feminism': 'Feminist Movement',
               'abortion': 'Legalization of Abortion',
               'atheism': 'Atheism'
               }

TOPICS_LONG_REVERSE = dict(zip(TOPICS_LONG.values(), TOPICS_LONG.keys()))


def istargetInTweet(devdata, target_list):
    """
    Check if target is contained in tweet
    :param devdata: development data as a dictionary (keys: targets, values: tweets)
    :param target_short: short version of target, e.g. 'trump', 'clinton'
    :param id: tweet number
    :return: true if target contained in tweet, false if not
    """
    cntr = 0
    ret_dict = {}
    for id in devdata.keys():

        tweet = devdata.get(id)
        target_keywords = KEYWORDS.get(TOPICS_LONG_REVERSE.get(target_list[0]))
        target_in_tweet = False
        for key in target_keywords:
            if key.lower() in tweet.lower():
                target_in_tweet = True
                break
        ret_dict[id] = target_in_tweet
        cntr += 1
    return ret_dict


def istargetInTweetSing(devdata, target_short):
    """
    Check if target is contained in tweet
    :param devdata: development data as a dictionary (keys: targets, values: tweets)
    :param target_short: short version of target, e.g. 'trump', 'clinton'
    :param id: tweet number
    :return: true if target contained in tweet, false if not
    """
    ret_dict = {}
    for id in devdata.keys():

        tweet = devdata.get(id)
        target_keywords = KEYWORDS.get(target_short)
        target_in_tweet = False
        for key in target_keywords:
            if key.lower() in tweet.lower():
                target_in_tweet = True
                break
        ret_dict[id] = target_in_tweet
    return ret_dict


def filterStopwords(tokenised_tweet, filter="all"):
    """
    Remove stopwords from tokenised tweet
    :param tokenised_tweet: tokenised tweet
    :return: tweet tokens without stopwords
    """
    if filter == "all":
        stops = stopwords.words("english")
        stops.extend(["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":",
                      ";", "<", ">", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~", "=", "+", "!",  "?"])
        stops.extend(["rt", "#semst", "...", "thats", "im", "'s", "via"])
    elif filter == "most":
        stops = []
        stops.extend(["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":",
                      ";", "<", ">", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~", "=", "+", "!", "?"])
        stops.extend(["rt", "#semst", "...", "thats", "im", "'s", "via"])
    elif filter == "punctonly":
        stops = []
        # extended with string.punctuation and rt and #semst, removing links further down
        stops.extend(["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":",
                  ";", "<", ">", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~"])  #"=", "+", "!",  "?"
        stops.extend(["rt", "#semst", "..."]) #"thats", "im", "'s", "via"])
    else:
        stops = ["rt", "#semst", "..."]

    stops = set(stops)
    return [w for w in tokenised_tweet if (not w in stops and not w.startswith("http"))]


def build_dataset(words, vocabulary_size=5000000, min_count=5):
    """
    Build vocabulary, code based on tensorflow/examples/tutorials/word2vec/word2vec_basic.py
    :param words: list of words in corpus
    :param vocabulary_size: max vocabulary size
    :param min_count: min count for words to be considered
    :return: counts, dictionary mapping words to indeces, reverse dictionary
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        if _ >= min_count:# or _ == -1:  # that's UNK only
            dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    print("Final vocab size:", len(dictionary))
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


def transform_tweet(w2vmodel, words, maxlen=20):
    """
    Transform list of tokens with word2vec model, add padding to maxlen
    :param w2vmodel: word2vec model
    :param words: list of tokens
    :param maxlen: maximum length
    :return: transformed tweet, as numpy array
    """
    data = list()
    for i in range(0, maxlen-1):  #range(0, len(words)-1):
        if i < len(words):
            word = words[i]
            if word in w2vmodel.vocab:
                index = w2vmodel.vocab[word].index
            else:
                index = w2vmodel.vocab["unk"].index
        else:
            index = w2vmodel.vocab["unk"].index
        data.append(index)
    return np.asarray(data)


def transform_tweet_dict(dictionary, words, maxlen=20):
    """
    Transform list of tokens, add padding to maxlen
    :param dictionary: dict which maps tokens to integer indices
    :param words: list of tokens
    :param maxlen: maximum length
    :return: transformed tweet, as numpy array
    """
    data = list()
    for i in range(0, maxlen-1):  #range(0, len(words)-1):
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



def tokenise_tweets(tweets, stopwords="all"):
    return [filterStopwords(tokenize(tweet.lower()), stopwords) for tweet in tweets]


def transform_targets(targets):
    ret = []
    for target in targets:
        if target == "Atheism":
            ret.append("#atheism")
        elif target == "Climate Change is a Real Concern":
            ret.append("#climatechange")
        elif target == "Feminist Movement":
            ret.append("#feminism")
        elif target == "Hillary Clinton":
            ret.append("#hillaryclinton")
        elif target == "Legalization of Abortion":
            ret.append("#prochoice")
        elif target == "Donald Trump":
            ret.append("#donaldtrump")
    return ret


def transform_labels(labels, dim=3):
    labels_t = []
    for lab in labels:
        v = np.zeros(dim)
        if dim == 3:
            if lab == 'NONE':
                ix = 0
            elif lab == 'AGAINST':
                ix = 1
            elif lab == 'FAVOR':
                ix = 2
        else:
            if lab == 'AGAINST':
                ix = 0
            elif lab == 'FAVOR':
                ix = 1
        v[ix] = 1
        labels_t.append(v)
    return labels_t


if __name__ == '__main__':
    tweets, targets, labels, ids = reader.readTweetsOfficial("../data/semeval2016-task6-train+dev.txt")
    tweet_tokens = tokenise_tweets(tweets)
    target_tokens = tokenise_tweets(targets)
    count, dictionary, reverse_dictionary = build_dataset([token for senttoks in tweet_tokens+target_tokens for token in senttoks])  #flatten tweets for vocab construction
    transformed_tweets = [transform_tweet_dict(dictionary, senttoks) for senttoks in tweet_tokens]
    transformed_targets = [transform_tweet_dict(dictionary, senttoks) for senttoks in target_tokens]
    transformed_labels = transform_labels(labels)
    print('Longest tweet', len(max(transformed_tweets,key=len)))
    print('Longest target', len(max(transformed_targets,key=len)))
    print('Most common words (+UNK)', count[:5])
    #print('Sample data', data[:10])
