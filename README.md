# Size Does Matter - Learning From Data Final Project

In this GitHub the code for the final project for Learning From Data can be found. This GitHub contains the dataset from Shared Task SemEval 2019 Task 6 (Sub-Task A, https://github.com/ZeyadZanaty/offenseval). Two extra datasets can also be found here, the 'short' dataset and the 'long' dataset. The short dataset contains tweets shorter than 80 characters and the long dataset contains tweets longer than 140 characters.

This GitHub does not include the GloVe embeddings that are used in this experiment. The GloVe embeddings can be found on https://nlp.stanford.edu/projects/glove/. The embeddings used are the Twitter embeddings from this site. I have only used the 25b and 50b Tweet embeddings because they seemed to work better on the LSTM with the parameters I used.

## Creating a Virtual Environment and Installing the Requirements
```
sudo apt install python3.10
sudo apt install python3.10-venv

python3.10 -m venv .venv

source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Python Files
There are three python files: base_lfd.py, lstm_lfd.py and llm_lfd.py. The three files include different models that are trained, the base_lfd.py file contains classic models, the lstm_lfd.py contains an lstm and the llm_lfd.py file contains the possibility to fine-tune large language models. It is recommended to run the llm_lfd.py file on GPU possibilities, like Google Colab or the Habròk server from the RUG.

The python files all have their own hyperparameter adjustments and possibilities. To see all possibilities you can use the -h hyperparameter after the base command like so:
```
python3 base_lfd.py -h
```
This gives you the output for the help command which looks like this for the command above:
```
usage: base_lfd.py [-h] [-t TRAIN_FILE] [-d DEV_FILE] [-te TEST_FILE] [-vec {bow,tfidf,both}]
                   [-ng NGRAM_RANGE NGRAM_RANGE] [-l] [-sw {english,bow_short,postgresql,None}] [-a ALPHA]
                   {svm,svml,rf} ...

positional arguments:
  {svm,svml,rf}         Choose the classifying algorithm to use
    svm                 Use Support Vector Machine as classifier
    svml                Use Linear kernel Support Vector Machine as classifier
    rf                  Use Random Forest algorithm as classifier

options:
  -h, --help            show this help message and exit
  -t TRAIN_FILE, --train_file TRAIN_FILE
                        Train file to learn from (default train.tsv)
  -d DEV_FILE, --dev_file DEV_FILE
                        Dev file to evaluate on (default dev.tsv)
  -te TEST_FILE, --test_file TEST_FILE
                        Test file to test the system prediction quality
  -vec {bow,tfidf,both}, --vectorizer {bow,tfidf,both}
                        Select vectorizer: bow (bag of words), tfidf or both
  -ng NGRAM_RANGE NGRAM_RANGE, --ngram_range NGRAM_RANGE NGRAM_RANGE
                        Set the ngram range, give two integers separated by space
  -l, --lemmas          Lemmatizes the tokenized data.
  -sw {english,bow_short,postgresql,None}, --stop_words {english,bow_short,postgresql,None}
                        Removes stop words from the texts, 'english' is the base by sklearn 'bow_short' and
                        'postgresql' give a custom list.
  -a ALPHA, --alpha ALPHA
                        Set the alpha for the base Naive Bayes classifier
```

This command can be executed on all three python files and shows all possible inputs for the files.

### Data analysis
The data_analysis.py file exists, but is mainly for myself; I added it for clarity but it just has some counters and other options to see what the data looks like. You have to hard-code in this which makes it not recommended to look at this.

