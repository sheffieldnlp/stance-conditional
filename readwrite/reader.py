import io
import json
import numpy as np

def readTweetsOfficial(tweetfile, encoding='windows-1252', tweetcolumn=2, topic="all"):
    """
    Read tweets from official files
    :param topic: which topic to use, if topic="all", data for all topics is read
    :param encoding: which encoding to use, official stance data is windows-1252 encoded
    :param tweetcolumn: which column contains the tweets
    :return: list of tweets, list of targets, list of labels
    """
    tweets = []
    targets = []
    labels = []
    ids = []
    for line in io.open(tweetfile, encoding=encoding, mode='r'):
        if line.startswith('ID\t'):
            continue
        if topic == "all":
            tweets.append(line.split("\t")[tweetcolumn])
            targets.append(line.split("\t")[tweetcolumn-1])
            lid = line.split("\t")[0]
            v = np.zeros(1)
            v[0] = lid
            ids.append(v)
            if tweetcolumn > 1:
                labels.append(line.split("\t")[tweetcolumn+1].strip("\n"))
            else:
                labels.append("UNKNOWN")
        elif topic in line.split("\t")[tweetcolumn-1].lower():
            tweets.append(line.split("\t")[tweetcolumn])
            targets.append(line.split("\t")[tweetcolumn-1])
            lid = line.split("\t")[0]
            v = np.zeros(1)
            v[0] = lid
            ids.append(v)
            if tweetcolumn > 1:
                labels.append(line.split("\t")[tweetcolumn+1].strip("\n"))
            else:
                labels.append("UNKNOWN")

    return tweets,targets,labels,ids



def readTweets(jsontweetfile):
    """
    Read tweets from json files
    :param jsontweetfile: file path of json file with tweets
    :return: list of tweets
    """
    tweets = []
    for line in open(jsontweetfile, 'r'):
        tweets.append(json.loads(line)['text'])
    return tweets