#
#
#
#

import argparse
import json
import logging
import random as python_random


import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)

# Setup logging configuration
logging.basicConfig(filename='/content/gdrive/MyDrive/FP_LFD/results_llm.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')


def read_corpus(corpus_file):
    '''Reads the given corpus, returns the documents and labels (OFF or NOT) in a list.'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            documents.append(" ".join(tokens[:-1]).strip())
            # 2-class problem: OFF vs NOT
            labels.append(tokens[-1])
    return documents, labels


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default="data/train.tsv", type=str,
                        help="Input file to learn from (default train.tsv)")
    parser.add_argument("-d", "--dev_file", type=str, default="data/dev.tsv",
                        help="Separate dev set to read in (default dev.tsv)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    parser.add_argument("-lr", "--learning_rate", default=0.0001, type=float,
                        help="Set the learning rate")
    parser.add_argument("-bs", "--batch_size", default=4, type=int,
                        help="Set the batch size")
    parser.add_argument("-ep", "--epochs", default=1, type=int,
                        help="Set the number of epochs")

    parser.add_argument("-tr", "--transformer", default=None, choices=["distilbert", "roberta", "electra", "deberta"])

    args = parser.parse_args()
    return args


# Custom function to log and print so we don't have to write all parameters
# down each run.
def log_and_print(message, printed=True):
    """Logs a message and prints it to the console."""
    logging.info(message)
    if printed:
        print(message)


def predict_transformers(model, tokens_dev, Y_dev_bin, ident):
    """
    Create prediction for the transformer model.
    """
    # Get predictions
    Y_pred_logits = model.predict(tokens_dev)["logits"]
    Y_pred_probs = tf.sigmoid(Y_pred_logits)
    Y_pred = (Y_pred_probs.numpy() > 0.5).astype(int)

    Y_test = Y_dev_bin

    log_and_print("Accuracy on own {1} set: {0}".format(round(accuracy_score(Y_test, Y_pred), 3), ident))
    log_and_print("f1 score on own {1} set: {0}".format(round(f1_score(Y_test, Y_pred, average="macro"), 3), ident))


def compile_transformer(lm, args):
    """
    Compile transformer model.
    """

    learning_rate = args.learning_rate

    # Num_labels is 1 as the prediction is made for the OFF class.
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=1)
    loss_function = BinaryCrossentropy(from_logits=True)
    optim = Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optim, metrics=["accuracy"])

    return model


def main():
    args = create_arg_parser()

    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    # Use train and dev to create vocab - could also do just train

    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)

    vectorizer.adapt(text_ds)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train) 
    Y_dev_bin = encoder.transform(Y_dev)

    if args.transformer == "electra":
        lm = "google/electra-small-discriminator"
    elif args.transformer == "roberta":
        lm = "FacebookAI/roberta-base"
    elif args.transformer == "deberta":
        lm = "microsoft/deberta-v3-base"
    else:
        lm = "distilbert/distilbert-base-cased"
        
    tokenizer = AutoTokenizer.from_pretrained(lm)
    transformer_tokens_train = tokenizer(X_train, padding=True, max_length=100,
                                         truncation=True, return_tensors="np").data
    
    transformer_tokens_dev = tokenizer(X_dev, padding=True, max_length=100,
                                       truncation=True, return_tensors="np").data

    model = compile_transformer(lm, args)
    model.fit(transformer_tokens_train, Y_train_bin, verbose=1, epochs=args.epochs,
              batch_size=args.batch_size, validation_data=(transformer_tokens_dev, Y_dev_bin))
    predict_transformers(model, transformer_tokens_dev, Y_dev_bin, "dev")

    # Do predictions on specified test set
    if args.test_file:
        # Read in test set and vectorize
        X_test, Y_test = read_corpus(args.test_file)
        Y_test_bin = encoder.transform(Y_test)
        transformer_tokens_test = tokenizer(X_test, padding=True, max_length=100,
                                            truncation=True, return_tensors="np").data
        # Finally do the predictions
        predict_transformers(model, transformer_tokens_test, Y_test_bin, "test")

    all_args = " \\\n".join([f" --{key}={value}" for key, value in vars(args).items() if value])
    log_and_print(f"Used settings:\n{all_args}", False)
    log_and_print(f"Model construction:\n{model.layers}\n", False)

if __name__ == "__main__":
    main()