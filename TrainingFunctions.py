import pandas as pd
import numpy as np
import glob
import pickle
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier as Neural
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from os.path import isfile


def compile_training_set(data_dir, save_file_int, save_file_class,
                         clobber=False, linelist='./data/linelabels'):
    """
    Constructs a data-frame containing the line flux values. The resulting DF
    is then saved as a pickle. The same is done with the class values

    Parameters
    ----------
    data_dir : str
        Directory from which to fetch the cloudy savefiles
    save_file_int : str
        Name of the file to which to save the pickled dataframe
    save_file_class : str
        Name of the file to which to save the classlabel dataframe
    linelist : str
        Path to the list containing the linelabels. Defaults to
        './data/linelabels'
    """
    done = isfile(save_file_int) and isfile(save_file_int) and clobber is False
    if done:
        print('Dataset already compiled')
        return

    # Get filenames
    flist = glob.glob(data_dir + '**/*.pkl')
    linenames = []
    with open(linelist, 'r') as f:
        lin = f.readlines()
        for l in lin:
            linenames.append(l.split('\n')[0])

    linedata = []
    classes = []
    for f in flist:
        print(f)
        data = pickle.load(open(f, 'rb'))
        lines = data['intensities']
        class_ = data['class']
        classes.append(class_)
        intensity = []
        for line_name in linenames:
            inten = lines[line_name]
            intensity.append(inten)
        linedata.append(intensity)

    result = pd.DataFrame(columns=linenames, data=linedata)
    response = pd.DataFrame()
    response['class'] = classes

    # Pickle the results
    result.to_pickle(save_file_int)
    response.to_pickle(save_file_class)


def add_noise(datafile, responsefile, overwrite=True, save_x=None,
              save_y=None):
    """
    Adds a gaussian noise component to the cloudy modelruns, effectively
    doubling the training set. Saves the resulting dfs as pickles

    Parameters
    ----------
    datafile : str
        file containing the main dataset pickle
    responsefile : str
        file containing the y classes
    overwrite : bool
        whether to overwrite the old files
    save_x : str
        If overwrite is false, this specifies the savelocation of the data
    save_y : str
        If overwrite is false, this specifies the savelocation of the responses

    """
    x_df = pd.read_pickle(datafile)
    y_df = pd.read_pickle(responsefile)
    second_y = y_df.copy()
    x_noise = x_df.apply(_add_single_noise)

    y_long = y_df.append(y_df).reset_index(drop=True)
    x_long = x_df.append(x_noise).reset_index(drop=True)

    if overwrite:
        x_long.to_pickle(datafile)
        y_long.to_pickle(responsefile)
    else:
        if save_x is not None:
            x_long.to_pickle(save_x)
        if save_y is not None:
            y_long.to_pickle(save_y)


def _add_single_noise(value):
    """
    Adds 10 percent gaussian noise to a number. Utility function for use in
    add_noise

    Parameters
    ----------
    value : float
        Value to which to add the noise

    Returns
    -------
    noise : float
        Value + 10% gaussian noise
    """
    number = np.random.normal(scale=0.1 * np.abs(value))
    return value + number


def load_data(x_file, y_file, standardize=True,
              standardizer_file='./data/standardizer.pkl'):
    """
    Loads and returns the data. If standardize is True (default) it
    standardizes the data and persists the standard scaler object for use on
    the actual dataset

    Parameters
    ----------
    x_file : str
        Name of the pickle containing the training data set.
    y_file : str
        Name of the pickle containing the training classes.
    standardize : bool
        Whether or not to standardize the data
    standardizer_file : str
        Name of file in which to persist the data

    Returns
    -------
    x_df : pandas.DataFrame
        training data set
    y_df : pandas.DataFrame
        training classes
    """
    x_df = pd.read_pickle(x_file)
    y_df = pd.read_pickle(y_file)
    if standardize:
        scaler = sklearn.preprocessing.StandardScaler()
        fit_scaler = scaler.fit(x_df)
        scaled_x_data = fit_scaler.transform(x_df)
        x_df_new = pd.DataFrame(columns=x_df.columns, data=scaled_x_data)

        # Persist the standardizer
        joblib.dump(fit_scaler, standardizer_file)
        return x_df_new, y_df
    return x_df, y_df


def train_model(data, response, model=None):
    """
    Trains the selected model on the full data set

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with all the measured line fluxes from CLoudy
    response : pandas.DataFrame
        Classes from Cloudy column density
    model : sklearn model object, optional
        An instance of an sklearn classifier. If not specified the function
        defaults to an AdaBoostClassifier

    Returns
    -------
    model : sklearn model object
        Returns the trained model
    """
    if model is None:
        model = AdaBoostClassifier()
        # model = Neural()
    resp = response.values.ravel()
    fitted_model = model.fit(data, resp)
    return fitted_model


def save_trained_model(model, filename):
    """
    Pickles the trained model

    Parameters
    ----------
    model : sklearn model object
        Trained model object
    file_name : str
        Name of the file to which to save the pickled model
    """
    joblib.dump(model, filename)


def cross_validate(data, response, model=None):
    """
    Run a k-fold cross validation on the model to assess predictive ability.

    Parameters
    ----------
    data : pandas.DataFrame
        The full training data set
    response : pandas.DataFrame
        The classes for the training set
    model : sklearn model (optional)
        Can be used  to specify another model to cross validate. Otherwise
        defaults to AdaBoostClassifier

    Returns
    -------
    scores : list
        List of all 5 scores.
    mean_score : float
        Mean of scores
    """
    data = data.copy().values
    resp = response.copy().values

    if model is None:
        model = AdaBoostClassifier()
    # Set up Kfold
    kf = KFold(n_splits=5)
    scores = []
    for train, test in kf.split(data):
        model.fit(data[train], resp[train].ravel())
        scores.append(model.score(data[test], resp[test].ravel()))
    return scores, np.array(scores).mean()
