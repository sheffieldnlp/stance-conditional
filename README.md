# stance-conditional

Experiments for "unseen target" stance detection with conditional encoding on the SemEval 2016 Stance Detection (http://alt.qcri.org/semeval2016/task6/) dataset.

Official stance data is available via https://www.dropbox.com/sh/o8789zsmpvy7bu3/AABRja7NDVPtbjSa-y3GH0jAa?dl=0  and collected unlabelled tweets are stored in https://www.dropbox.com/sh/7i2zdnet49yb1sh/AAA_AzN64JLuNlfU5pt69W8ia?dl=0

- semeval2016-task6-trainingdata.txt is the official training set for Task A (seen target)
- semeval2016-task6-trialdata.txt is the official trial data for Task A (seen target)
- semeval2016-task6-train+dev.txt is the official training data concatenated with the official trial data for Task A (seen target)
- semEval2016-Task6-subtaskA-testdata-gold.txt is the official test data for Task A (seen target)
- downloaded_Donald_Trump.txt is the official development data for Task B (unseen target)
- semEval2016-Task6-subtaskB-testdata.txt is the official test data for Task B (unseen target)
- additionalTweetsStanceDetection.json contains additional crawled tweets (November - January) containing, each mentioning at least one of the targets for Task A or Task B
- NEW: additionalTweetsStanceDetectionBig.json contains a bit more additional crawled tweets (November - February) containing, each mentioning at least one of the targets for Task A or Task B

Note that for the unseen target task, no labelled training data is available, only unlabelled development data. The Task A train + dev data can be used for training instead.

Current data sizes:

- Unlabelled crawled tweets: 395212
- Donald Trump tweets: 16692  
- Official labelled training tweets: 44389  
- Overall 129887 tokens (25166072 tokens including singletons)

Results:

- Official results for Task A and Task B are available here: https://docs.google.com/spreadsheets/d/11k0XKaYwJ-Xh-C9sm5M_a6kfQJCsK0PlPTLJs1dDZrU/edit#gid=0
- A visualisation of the official train and test datasets by the task organisers is available here: http://www.saifmohammad.com/WebPages/StanceDataset.htm

Dependencies:

- Python 3.5
- Numpy
- Tensorflow 0.6 (https://www.tensorflow.org/versions/master/get_started/os_setup.html#pip-installation)

Running instructions:

- Download all the above mentioned files and save them in the data/ subfolder
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
- To run stance detection training and test with conditional encoding:
```shell
python3 conditional.py
```
