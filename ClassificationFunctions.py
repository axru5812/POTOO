import pandas as pd
import numpy as np
import glob
import pickle
import os
import sklearn
from sklearn.externals import joblib


def compile_data_set(file_dir='./data/lines/', save_name='./data/sdss_df.pkl',
                     clobber=False):
    """
    Compiles the SDSS data set

    Parameters
    ----------
    file_dir : str
        Directory to the individual SDSS line data files
    save_name : str
        Name of the savefile
    clobber : bool
        Whether to overwrite an already compiled file
    """
    if os.path.isfile(save_name) and clobber is False:
        print('Dataset already compiled')
    else:
        flist = glob.glob(file_dir + '*')
        store = []
        line_order = ['H  1  4861.33A',
                      'H  1  6562.81A',
                      'Ca B  5875.64A',
                      'N  2  6583.45A',
                      'O  1  6300.30A',
                      'O  2  3726.03A',
                      'O  2  3728.81A',
                      'O  3  5006.84A',
                      'BLND  4363.00A',
                      'S  2  6716.44A',
                      'S  2  6730.82A',
                      'Cl 3  5517.71A',
                      'Cl 3  5537.87A']

        for f in flist:
            df = pd.read_pickle(f)
            name = [f.split('.')[0]]
            new_df = pd.DataFrame(data=df['flux'].values, index=line_order,
                                  columns=name).T
            store.append(new_df)

        # Create the monster
        monster = pd.concat(store)
        monster.to_pickle(save_name)


def load_data(filename='./data/sdss_df.pkl', standardize=True,
              standardizer_file='./data/standardizer.pkl'):
    data = pd.read_pickle(filename)
    if standardize:
        standardizer = joblib.load(standardizer_file)
        new_data = standardizer.transform(data)
        data_df = pd.DataFrame(columns=data.columns, index=data.index,
                               data=new_data)
        return data_df
    return data


def load_model(filename):
    """
    Loads a trained model instance

    Parameters
    ----------
    filename : str
        Name of pickle file to load.

    Returns
    -------
    model : sklearn model object
        Returns the trained sklearn model.
    """
    model = joblib.load(filename)
    return model


def classify(model, data):
    """
    Runs the classifier and returns the predicted classes

    Parameters
    ----------
    model : sklearn model object

    Returns
    -------
    classes : pd.DataFrame
        Dataframe containing all the predicted classes
    classA : pd.DataFrame
        Dataframe with the results that have class case A
    percent : float
        Percentage of the spectra which are found to match column densities < 17.2
    """
    classes = model.predict(data)
    classA = pd.DataFrame(index=data.index, data=classes)
    classA.drop(data.index[np.where(classes != 'A')], inplace=True)

    percent = (len(classA) / len(classes)) * 100

    return classes, classA, percent
