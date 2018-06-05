"""
Script that runs the classification.
"""
import pandas as pd
import numpy as np
import ClassificationFunctions as c
import matplotlib.pyplot as plt
from PythonUtilities.graphics import plotconfig


def main():
    print('Compiling the data')
    c.compile_data_set()
    print('Loading datafiles into memory')
    data = c.load_data()
    print('\tLoaded {} galaxies'.format(len(data.index)))
    print('Loading the trained model')
    model = c.load_model('./data/trained_model.pkl')
    print('\tLoaded: {}'.format(model))
    print('Classifying the galaxies')
    classes, classA_galaxies, percent = c.classify(model, data)
    print('\tNumber of case A galaxies: ', len(classA_galaxies.index))
    print('\tPercentage of case A galaxies: ', percent)


if __name__ == '__main__':
    main()
