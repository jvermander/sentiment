#!/usr/bin/env python3.7
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

def main( argv ):
  clf = pickle.load(open('model.sav', 'rb'))
  output = clf.predict([argv[1]])
  print(output[0])


if(__name__ == "__main__"):
  main(sys.argv)