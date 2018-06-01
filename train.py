import pandas as pd
import numpy as np
import TrainingFunctions as t


def main():
    print('Compiling training set')
    t.compile_training_set('./data/cloudy/results/',
                           './data/training_data.pkl',
                           './data/training_class.pkl')
    print('Loading data')
    data, response = t.load_data('./data/training_data.pkl',
                                 './data/training_class.pkl')
    print('Training model')
    model = t.train_model(data, response)
    print('Running Cross validation')
    scores, mean_score = t.cross_validate(data, response)
    print('\tCross validation score', mean_score)
    t.save_trained_model(model, './data/trained_model.pkl')


if __name__ == '__main__':
    main()
