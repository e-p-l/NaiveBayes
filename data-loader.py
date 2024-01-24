import numpy as np
import pandas as pd
import matplotlib as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer

# train['text'] = list of all the text
# train['label'] = list of all the labels
def get_emotion_data(range, num_of_features, remove_stop_words):
    test = load_dataset("dair-ai/emotion", split="test")
    train = load_dataset("dair-ai/emotion", split="train")
    if(remove_stop_words == True):
      train_vectorizer = CountVectorizer(ngram_range=range, max_features=num_of_features, stop_words="english")
      train_bow_features = train_vectorizer.fit_transform(train['text']).toarray()
      train_labels = train['label']
      train_vocabulary = train_vectorizer.get_feature_names_out()

      test_vectorizer = CountVectorizer(ngram_range=range, vocabulary=train_vocabulary, stop_words="english")
      test_bow_features = test_vectorizer.fit_transform(test['text']).toarray()
      test_labels = test['label']
    else:
      train_vectorizer = CountVectorizer(ngram_range=range, max_features=num_of_features, stop_words=None)
      train_bow_features = train_vectorizer.fit_transform(train['text']).toarray()
      train_labels = train['label']
      train_vocabulary = train_vectorizer.get_feature_names_out()

      test_vectorizer = CountVectorizer(ngram_range=range, vocabulary=train_vocabulary, stop_words=None)
      test_bow_features = test_vectorizer.fit_transform(test['text']).toarray()
      test_labels = test['label']

    return (train_bow_features, train_labels), (test_bow_features, test_labels)