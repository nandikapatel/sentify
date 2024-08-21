import random
import re
import string

import nltk
from nltk import FreqDist, classify, NaiveBayesClassifier
from nltk.corpus import twitter_samples, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def remove_noise(tokenized_sentences, stopping_words=()):
    cleaned_tokens = []

    for token, tag in pos_tag(tokenized_sentences):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*(),]|'
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatize = WordNetLemmatizer()
        token = lemmatize.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stopping_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_all_words(cleaned_tokens_list):
    for cleaned_tokens in cleaned_tokens_list:
        for token in cleaned_tokens:
            yield token


def get_words_for_model(cleaned_tokens_list):
    for sentences_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in sentences_tokens)


if __name__ == "__main__":

    positive_words = twitter_samples.strings('positive-words-final.json')
    negative_words = twitter_samples.strings('negative-words-final.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    sentence_tokens = twitter_samples.tokenized('positive-words-final.json')[0]
    stop_words = stopwords.words('english')

    positive_sentence_tokens = twitter_samples.tokenized('positive-words-final.json')
    negative_sentence_tokens = twitter_samples.tokenized('negative-words-final.json')
    # print(positive_sentence_tokens)
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_sentence_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_sentence_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    freq_dist_pos = FreqDist(all_pos_words)
    print(freq_dist_pos.most_common(10))

    positive_tokens_for_model = get_words_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_words_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset[:50000]
    test_data = dataset[50000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))

    custom_sentence = "the beach is fun"

    custom_tokens = remove_noise(word_tokenize(custom_sentence))

    print(custom_sentence, classifier.classify(dict([token, True] for token in custom_tokens)))
# NNP: Noun, proper, singular
# NN: Noun, common, singular or mass
# IN: Preposition or conjunction, subordinating
# VBG: Verb, gerund or present participle
# VBN: Verb, past participle
