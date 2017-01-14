# stance-conditional

Experiments for "unseen target" stance detection with conditional encoding on the SemEval 2016 Stance Detection (http://alt.qcri.org/semeval2016/task6/) dataset.

Paper: Isabelle Augenstein, Tim Rocktäschel, Andreas Vlachos, Kalina Bontcheva. Twitter Stance Detection with Bidirectional Conditional Encoding. Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2016), November 2016.

Official stance data is available via https://www.dropbox.com/sh/o8789zsmpvy7bu3/AABRja7NDVPtbjSa-y3GH0jAa?dl=0  and collected unlabelled tweets are stored in https://www.dropbox.com/sh/7i2zdnet49yb1sh/AAA_AzN64JLuNlfU5pt69W8ia?dl=0

- semeval2016-task6-trainingdata.txt is the official training set for Task A (seen target)
- semeval2016-task6-trialdata.txt is the official trial data for Task A (seen target)
- semeval2016-task6-train+dev.txt is the official training data concatenated with the official trial data for Task A (seen target)
- semEval2016-Task6-subtaskA-testdata-gold.txt is the official test data for Task A (seen target)
- downloaded_Donald_Trump.txt is the official development data for Task B (unseen target)
- semEval2016-Task6-subtaskB-testdata.txt is the official test data for Task B (unseen target)
- additionalTweetsStanceDetection.json contains additional crawled tweets (November - January) containing, each mentioning at least one of the targets for Task A or Task B

Note that for the unseen target task, no labelled training data is available, only unlabelled development data. The Task A train + dev data can be used for training instead.


Results:

- Official results for Task A and Task B are available here: https://docs.google.com/spreadsheets/d/11k0XKaYwJ-Xh-C9sm5M_a6kfQJCsK0PlPTLJs1dDZrU/edit#gid=0
- A visualisation of the official train and test datasets by the task organisers is available here: http://www.saifmohammad.com/WebPages/StanceDataset.htm

Dependencies:

- Python 3.5
- Numpy
- sklearn
- Gensim
- Tensorflow 0.6 (https://www.tensorflow.org/versions/master/get_started/os_setup.html#pip-installation). To get version 0.6 on Mac OS use: `pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.6.0-py3-none-any.whl`
- nltk

Running instructions:

- Download all the above mentioned files and save them in the data/ subfolder
- Download the official SemEval evaluation materials from [here](http://alt.qcri.org/semeval2016/task6/data/uploads/eval_semeval16_task6_v2.zip) and put the file `eval.pl` in the `stancedetection` directory.
- Add the twokenize submodule: 
```
git submodule add https://github.com/leondz/twokenize.git twokenize_wrapper
touch twokenize_wrapper/__init__.py
```
- Make sure twokenize_wrapper, readwrite, and stancedetection are on PYTHONPATH
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

Note that accuracy and loss are printed during training for the training set, whereas P/R/F1 are printed during training for the dev or test set.


Bidirectional Conditional Encoding for Tensorflow version 0.11+:

- A version of the bidirectional conditional reader for Tensorflow version 0.11+ implemented with dynamic RNNs is contained in ```stancedetection/bicond_tf11.py```
- Note that this is only for illustration purposes / for those who want to apply the model to other tasks, and it is not connected to the rest of the code
