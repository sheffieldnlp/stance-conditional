# stance-conditional

Experiments for "unseen target" stance detection with conditional encoding on the SemEval 2016 Stance Detection (http://alt.qcri.org/semeval2016/task6/) dataset.

Official stance data is available via https://www.dropbox.com/sh/o8789zsmpvy7bu3/AABRja7NDVPtbjSa-y3GH0jAa?dl=0, collected unlabelled tweets and word2vec models are stored in https://www.dropbox.com/sh/7i2zdnet49yb1sh/AAA_AzN64JLuNlfU5pt69W8ia?dl=0

- semeval2016-task6-trainingdata.txt is the official training set for Task A (seen target)
- semeval2016-task6-trialdata.txt is the official trial data for Task A (seen target)
- semeval2016-task6-train+dev.txt is the official training data concatenated with the official trial data for Task A (seen target)
- semeval2016-task6-train+dev_*.txt is the official training data concatenated with the official trial data for Task A, split by target (seen target)
- semEval2016-Task6-subtaskA-testdata-gold.txt is the official test data for Task A (seen target)
- semEval2016-Task6-subtaskA-testdata-gold_*.txt is the official test data for Task A, split by target (seen target)
- downloaded_Donald_Trump_all.txt is the official development data for Task B (unseen target)
- semEval2016-Task6-subtaskB-testdata.txt is the official test data for Task B (unseen target)
- additionalTweetsStanceDetection.json contains additional crawled tweets (November - January) containing, each mentioning at least one of the targets for Task A or Task B
- additionalTweetsStanceDetectionBig.json contains a bit more additional crawled tweets (November - February) containing, each mentioning at least one of the targets for Task A or Task B (not used)
- NEW: trump_autolabelled_morehashs.txt contains automatically labelled Donald Trump tweets

Note that for the unseen target task, no labelled training data is available, only unlabelled development data. The Task A train + dev data can be used for training instead.

Current data sizes:

- Unlabelled crawled tweets: 395212
- Donald Trump tweets: 278013  
- Official labelled training tweets: 44389  
- Automatically labelled Donald Trump tweets: 9776

Results:

- Official results for Task A and Task B are available here: https://docs.google.com/spreadsheets/d/11k0XKaYwJ-Xh-C9sm5M_a6kfQJCsK0PlPTLJs1dDZrU/edit#gid=0
- A visualisation of the official train and test datasets by the task organisers is available here: http://www.saifmohammad.com/WebPages/StanceDataset.htm

Dependencies:

- Python 3.5
- Numpy
- Tensorflow 0.6 (https://www.tensorflow.org/versions/master/get_started/os_setup.html#pip-installation)

Running instructions:

- Download all the above mentioned training and testing files and save them in the data/ subfolder
- Download the word2vec models and save them in the out/ folder
- In ```conditional.py```, change ```sys.path.append('/Users/Isabelle/stance-conditional/')``` to correct path on your system
- If not already there, add ssh key to github account (https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/)
- Fetch tfrnn and naga submodule code:
```shell
git submodule update --init --recursive
```
- Make sure tfrnn-repo/naga-repo, twokenize, readwrite, and stancedetection are on PYTHONPATH
- Run word2vec_training.py first to pre-train a word2vec model on the unlabelled data
```shell
mkdir out
cd stancedetection
python3 word2vec_training.py
```
- To run stance detection training and test with conditional encoding and default settings:
```shell
python3 conditional.py
```

Note that accuracy and loss are printed during training for the training set, whereas P/R/F1 are printed during training for the dev or test set.

- Current parameters of ```conditional.py``` with possible values in (TODO: once those are finalised, change them to command line parameters):
  - SINGLE_RUN = False  # only one run
  - EVALONLY = False   # evaluate a file again using the eval script
  - TESTONLY = False   # load a pre-trained model. Assumes there exists a pre-trained model with the exact configurations listed below.
                       # A model is saved every two epochs, so this is useful for recovering a model from a previous epoch.
  - testrange = (0, 9)  # how many runs, those are the test IDs
  - input_size = 100  # note that a word2vec model of the same size has to be available. Currently available: 100, 200
  - hidden_size = [100] # anything < 200. Sensible values: 60, 100, 150, 200. Add several different ones to the list to test what works best.
  - modeltype = ["experimental"] # this is the currently best model. Other options: "conditional-target-feed", "conditional-reverse", "conditional", "aggregated", "tweetonly"
  - word2vecmodel = ["skip_nostop_sing_100features_5minwords_5context_big"] # others available: "skip_nostop_sing_200features_5minwords_5context_big", "default"
  - dropout = ["true"] # This sets the dropout to 0.1. Others available: "false", though it might make sense to experiment with higher rates of dropout.
  - testsetting = ["true"] # Others available: "false" (which is the dev setting using Hillary Clinton tweets), "taskA", "taskA-*" (for only using a certain target for training and testing. Abbreviations have to match the file names, i.e. "a", "cc", "la", "fm", "hc")
  - pretrain = ["pre_cont"] # use pre-trained word embeddings and continue training. Other options: "false" (initialise randomly), "pre" (use pre-trained word embeddings but do not continue training them)
  - num_epochs = 20
