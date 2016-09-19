import io
from readwrite import reader, writer
import preprocess

def selectTrainData(tweets, targets):
    inv_topics = {v: k for k, v in preprocess.TOPICS_LONG.items()}
    inlist = []
    outcnt = 0
    for i, tweet in enumerate(tweets):
        target_keywords = preprocess.KEYWORDS.get(inv_topics.get(targets[i]))
        target_in_tweet = 0
        for key in target_keywords:
            if key.lower() in tweet.lower():
                target_in_tweet = 1
                break
        if target_in_tweet == 1:
            inlist.append(i)
        else:
            outcnt += 1
    print("Incnt", len(inlist), "Outcnt", outcnt)
    return inlist


def printInOutFiles(inlist, infile, outfileIn, outfileOut):
    outfIn = open(outfileIn, 'w')
    outfOut = open(outfileOut, 'w')
    cntr = 0
    for line in io.open(infile, encoding='windows-1252', mode='r'):  # for the Trump file it's utf-8
        if line.startswith('ID\t'):
            outfIn.write(line)
            outfOut.write(line)
        else:
            if cntr in inlist:
                outfIn.write(line)
            else:
                outfOut.write(line)
            cntr += 1

    outfIn.close()
    outfOut.close()


if __name__ == '__main__':
    testdata = "../data/SemEval2016-Task6-subtaskB-testdata-gold.txt"
    devdata = "../data/semEval2016-task6-trialdata_new.txt"
    traindata = "../data/semeval2016-task6-train+dev.txt"

    devbest = "../out/results_all-1e-3-false_conditional-reverse_w2vsmall_hidd60_droptrue_stop-most_pre_cont_accthresh0.98_2.txt"

    tweets_gold, targets_gold, labels_gold, ids_gold = reader.readTweetsOfficial(devdata, 'windows-1252', 2)
    tweets_res, targets_res, labels_res, ids_res = reader.readTweetsOfficial(devdata, 'windows-1252', 2)

    inlist = selectTrainData(tweets_gold, targets_gold)

    printInOutFiles(inlist, devbest, "out_dev_inTwe_cond.txt", "out_dev_outTwe_cond.txt")
    printInOutFiles(inlist, devdata, "_gold_dev_inTwe.txt", "_gold_dev_outTwe.txt")

    print("All")
    writer.eval(devdata, devbest)

    print("Inlist")
    writer.eval("_gold_dev_inTwe.txt", "out_dev_inTwe_cond.txt")

    print("Outlist")
    writer.eval("_gold_dev_outTwe.txt", "out_dev_outTwe_cond.txt")