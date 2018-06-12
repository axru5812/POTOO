"""
Script that does the training of model.

Usage: python train.py

Requirements: Correctly set cloudy directory. Existing directory called ./data
for saving of model

Model is saved in a pickle in the data directory for easy loading in the
classification.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier as Neural
import TrainingFunctions as t


def main():
    print('Compiling training set')
    t.compile_training_set('./data/cloudy/results/',
                           './data/training_data.pkl',
                           './data/training_class.pkl')
    print('Add noise')
    t.add_noise('./data/training_data.pkl', './data/training_class.pkl')
    print('Loading data')
    data, response = t.load_data('./data/training_data.pkl',
                                 './data/training_class.pkl')
    print('Training model')
    model = t.train_model(data, response)
    print('Running Cross validation')
    scores, mean_score = t.cross_validate(data, response)
    print('\tPercent of samples correctly classified: ', mean_score * 100)
    t.save_trained_model(model, './data/trained_model.pkl')


if __name__ == '__main__':
    main()
