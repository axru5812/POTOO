import SDSSmanagement as sdss
import pandas as pd
import numpy as np
from astropy import coordinates as coords
from astropy import units as u
from astropy import table
from astropy.io import ascii as save_asc
import glob
import pickle


def construct_training_set(data_dir, save_file):
    """
    Constructs a data-frame containing the line flux values. The resulting DF
    is then saved as a pickle. The same is done with the class values

    Parameters
    ----------
    data_dir : str
        Directory from which to fetch the cloudy savefiles
    save_file : str
        Name of the file to which to save the pickled dataframe
    """
    # Get filenames
    flist = glob.glob('cloudy/**/*')
    for f in flist:
        data = pickle.load(f)
        f.

def save_trained_model(model, file_name):
    """
    Pickles the trained model

    Parameters
    ----------
    model : sklearn model object
        Trained model object
    file_name : str
        Name of the file to which to save the pickled model
    """
    pass


def train_model(model, data, response):
    """
    Trains the selected model on the full data set

    Parameters
    ----------
    model : sklearn model object
        eg. sklean.linear_model.linearRegression or similar
    data : pandas.DataFrame
        Dataframe with all the measured line fluxes from CLoudy
    response : pandas.DataFrame
        Classes from Cloudy column density

    Returns
    -------
    model : sklearn model object
        Returns the trained model
    """

def cross_validate(model, data, response):
    """
    Run a k-fold cross validation on the model to assess predictive ability.

    """
