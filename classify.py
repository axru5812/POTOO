import pandas as pd
import numpy as np
import ClassificationFunctions as c


def main():
    print('Compiling the data')
    c.compile_data_set()
    print('Loading datafiles into memory')
    data = c.load_data()
    print('Loading the trained model')
    model = c.load_model('./data/trained_model.pkl')
    print('Classifying the galaxies')
    classes, classA_galaxies, percent = c.classify(model, data)
    print('\tPercentage of case A galaxies: ', percent)


if __name__ == '__main__':
    main()
