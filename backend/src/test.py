# Sentiment prediction
# Dataset: kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format

# coords: x:600, y:250, w:475, h:700

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

import sys
import string
import pickle

def main():
  data = pd.read_csv('csv/data.csv')

  x = data['text']
  y = data['label']

  model = build_model()
  # nested_cross_validation(model, x, y)
  cross_validation(model, x, y)
  train_model(model, x, y)


def train_model( clf, x, y ):
  clf.fit(x, y)
  pickle.dump(clf, open('model.sav', 'wb'))


def build_model():
  rand = None
  grid = {
    'neural__hidden_layer_sizes': [(64, 32, 16)], 
    'vocab__max_features': [5000],
    'neural__alpha': [0.01]
  }
  pipe = Pipeline(
    [('vocab', CountVectorizer(stop_words='english', max_features=15000)),
     ('neural', MLPClassifier(hidden_layer_sizes=(64,32,32,32,16), 
                              activation='relu', 
                              alpha=3, 
                              early_stopping=True, 
                              random_state=rand))
  ])
  # cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rand)
  # clf = GridSearchCV(estimator=pipe, param_grid=grid, cv=cv, verbose=0, n_jobs=-1)
  clf = pipe
  return clf

def nested_cross_validation( clf, x, y ):
  skf = StratifiedKFold(n_splits=3)
  for train_ix, test_ix in skf.split(x, y):
    x_train, y_train = x[train_ix], y[train_ix]
    x_test, y_test = x[test_ix], y[test_ix]
    clf.fit(x_train, y_train)
    print('Params: ', clf.best_params_)
    print('Inner: ', clf.best_score_)
    print('Outer:', clf.score(x_test, y_test))

def cross_validation( clf, x, y ):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y)
  clf.fit(x_train, y_train)
  print('Training accuracy: ', clf.score(x_train, y_train))
  print('Test accuracy: ', clf.score(x_test, y_test))

if(__name__ == "__main__"):
  main()