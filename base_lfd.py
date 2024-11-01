# Name: base_lfd.py
#
#
#

import argparse
import random
import logging

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, f1_score
from nltk.tokenize import TweetTokenizer

random.seed(10)

logging.basicConfig(filename='results_base.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')


def check_valid_gamma(value):
    """
    Check if the given value for gamma is valid.

    Arguments:
        value (str): The given gamma value.

    returns:
        string | float: The given gamma value.
    """
    if value in ['scale', 'auto']:
        return value

    try:
        float_value = float(value)
        if float_value > 0.0:
            return float_value
        else:
            raise argparse.ArgumentTypeError(f"Given gamma: {value} is invalid, should be positive.")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Given gamma: {value} is invalid, should be float or 'scale' or 'auto'.")


def create_arg_parser():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-t", "--train_file", default='data/train.tsv', type=str,
                        help="Train file to learn from (default train.tsv)")
    parser.add_argument("-d", "--dev_file", default='data/dev.tsv', type=str,
                        help="Dev file to evaluate on (default dev.tsv)")
    parser.add_argument("-te", "--test_file", type=str,
                        help="Test file to test the system prediction quality")
    parser.add_argument("-vec", "--vectorizer", choices=["bow", "tfidf", "both"],
                        default="bow", help="Select vectorizer: bow (bag of words), tfidf or both")

    parser.add_argument("-ng", "--ngram_range", nargs=2, type=int, default=(1, 1),
                        help="Set the ngram range, give two integers separated by space")
    parser.add_argument("-l", "--lemmas", action="store_true",
                        help="Lemmatizes the tokenized data.")
    parser.add_argument("-sw", "--stop_words", choices=["english", "bow_short", "postgresql", None], default=None,
                        help="Removes stop words from the texts, 'english' is the base by sklearn "
                        "'bow_short' and 'postgresql' give a custom list.")

    parser.add_argument("-a", "--alpha", default=1.0, type=float,
                        help="Set the alpha for the base Naive Bayes classifier")
    
    subparser = parser.add_subparsers(dest="algorithm", required=False,
                                      help="Choose the classifying algorithm to use")
    
    svm_parser = subparser.add_parser("svm",
                                      help="Use Support Vector Machine as classifier")
    svm_parser.add_argument("-c", "--C", default=1.0, type=float,
                            help="Set the regularization parameter C of SVM")
    svm_parser.add_argument("-g", "--gamma", default='scale', type=check_valid_gamma,
                            help="Set gamma value, can be scale, auto or positive float.")

    svm_parser.add_argument("-cw", "--class_weight", choices=['balanced', None], default=None)
    svm_parser.add_argument("-k", "--kernel", choices=["linear", "poly", "rbf", "sigmoid"],
                            default="rbf",
                            help="Set the kernel, linear is already used by Linear SVM")
    svm_parser.add_argument("-dg", "--degree", default=3, type=int,
                            help="Set the degree, only works for poly kernel")

    # Subparser for Linear SVM
    svml_parser = subparser.add_parser("svml",
                                       help="Use Linear kernel Support Vector Machine as classifier")
    svml_parser.add_argument("-c", "--C", default=1.0, type=float,
                             help="Set the regularization parameter C of Linear SVM")
    svml_parser.add_argument("-cw", "--class_weight", choices=['balanced', None], default=None)
    svml_parser.add_argument("-p", "--penalty", choices=["l1", "l2"], default="l2",
                             help="Set the penalty parameter for Linear SVM")
    svml_parser.add_argument("-l", "--loss", choices=["hinge", "squared_hinge"], default="squared_hinge",
                             help="Set the loss parameter for Linear SVM, using hinge and penalty l1 is not supported "
                                  "by model")
    
    # Subparser for Random Forest
    forest_parser = subparser.add_parser("rf",
                                       help="Use Random Forest algorithm as classifier")
    forest_parser.add_argument("-ne", "--number_estimators", default=100, type=int,
                               help="Set the number of estimators of the forest")
    forest_parser.add_argument("-c", "--criterion", choices=["gini", "entropy", "log_loss"],
                               default="gini", help="Assign the fuction to measure the split quality")
    forest_parser.add_argument("-md", "--max_depth", default=None, type=int,
                               help="Set the maximum depth of the forest")
    forest_parser.add_argument("-mss", "--min_samples_split", default=2, type=int,
                               help="Set the minimum number of samples required to split an internal node")
    forest_parser.add_argument("-msl", "--min_samples_leaf", default=1, type=int,
                               help="Set the minimum number of samples per leaf node")

    args = parser.parse_args()
    return args


def log_and_print(message, printed=True):
    """Logs a message and prints it to the console."""
    logging.info(message)
    if printed:
        print(message)


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


def find_stop_words(arguments):
    '''If stopwords are requested finds the stop words lists'''

    sw_list = []

    if arguments.stop_words == 'bow_short':
        with open('stopwords/bow_short_sw.txt') as file:
            for line in file:
                sw_list.append(line.strip())

        return sw_list

    if arguments.stop_words == 'postgresql':
        with open('stopwords/postgresql_sw.txt') as file:
            for line in file:
                sw_list.append(line.strip())

        return sw_list

    return 'english'


def select_vectorizer(arguments):
    """
    Initialize the vectorizer based on the given arguments.
    """
    # Initialize vectorizers with selected arguments.
    sw = arguments.stop_words

    if sw is not None:
        sw = find_stop_words(arguments)

    tf_idf = TfidfVectorizer(preprocessor=identity, tokenizer=identity, ngram_range=tuple(arguments.ngram_range), stop_words=sw)
    bow = CountVectorizer(preprocessor=identity, tokenizer=identity, ngram_range=tuple(arguments.ngram_range), stop_words=sw)
    union = FeatureUnion([("count", bow), ("tf", tf_idf)])

    if arguments.vectorizer == "tfidf":
        # TF-IDF vectorizer
        return tf_idf
    elif arguments.vectorizer == "bow":
        # Bag of Words vectorizer
        return bow
    elif arguments.vectorizer == "both":
        # Use BoW and TF-IDF vectorizers.
        return union


def select_classifier(arguments):
    """
    Select the classifier and initialize it with the given arguments.
    """
    algorithm = ComplementNB(alpha=arguments.alpha)


    if arguments.algorithm == "svm":
        algorithm = SVC(C=arguments.C, gamma=arguments.gamma,
                        kernel=arguments.kernel, class_weight=arguments.class_weight,
                        degree=arguments.degree)
    
    if arguments.algorithm == "svml":
        algorithm = LinearSVC(C=arguments.C, penalty=arguments.penalty,
                              loss=arguments.loss, class_weight=arguments.class_weight, random_state=10)

    if arguments.algorithm == "dt":
        algorithm = RandomForestClassifier(criterion=arguments.criterion, max_depth=arguments.max_depth,
                                           n_estimators=arguments.number_estimators,
                                           min_samples_split=arguments.min_samples_split,
                                           min_samples_leaf=arguments.min_samples_leaf, random_state=10)

    return algorithm


def identity(inp):
    '''Dummy function that just returns the input, if lemmatize is a given argument
    return a lemmatized version of the input.
    '''

    tokenizer = TweetTokenizer()
    tokenized = [" ".join(tokenizer.tokenize(word)) for word in inp]

    if args.lemmas:
        lemmatizer = WordNetLemmatizer()
        lemma_list = [lemmatizer.lemmatize(word) for word in tokenized]

        return lemma_list
    
    return tokenized


if __name__ == "__main__":
    args = create_arg_parser()

    # Obtains the train features and labels.
    X_train, Y_train = read_corpus(args.train_file)
    # Select either test or dev set for evaluation and generate features and labels.
    X_test, Y_test = read_corpus(args.dev_file)

    # Uncomment this to see the label distribution for the training dataset.
    # print(Counter(Y_train))

    # Initializes the vectorizer
    vec = select_vectorizer(args)

    # Initializes the algorithm
    algorithm = select_classifier(args)

    classifier = Pipeline([('vec', vec), ('cls', algorithm)])

    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average="macro")

    log_and_print(f"Final accuracy on the Development set: {round(acc, 3)}")
    log_and_print(f"Macro F1-score on the Development set: {round(f1, 3)}")

    if args.test_file:
        X_test, Y_test = read_corpus(args.test_file)
        vec = select_vectorizer(args)
        # Initializes the algorithm
        algorithm = select_classifier(args)

        classifier = Pipeline([('vec', vec), ('cls', algorithm)])

        classifier.fit(X_train, Y_train)

        Y_pred = classifier.predict(X_test)

        acc = accuracy_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred, average="macro")

        log_and_print(f"Final accuracy on the Test set: {round(acc, 3)}")
        log_and_print(f"Macro F1-score on the Test set: {round(f1, 3)}")
    
    all_args = " \\\n".join([f" --{key}={value}" for key, value in vars(args).items() if value])
    log_and_print(f"Used settings:\n{all_args}\n", False)
