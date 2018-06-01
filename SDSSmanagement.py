import pandas as pd
import numpy as np
from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy import units as u
from astropy import table
from astropy.io import ascii as save_asc
from astropy.io import fits
import os
import sys
import scipy.signal as signal
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import copy
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def make_SDSS_idlist(filename, clobber=False, verbose=True):
    """
    Creates a master file with quantities that provide unique identification in
    the SDSS: ra, dec, mjd, plate number, and fiber ID.
    The spectra are selected to be galaxies and classified as starbursts
    => EW(Ha) > 50 Å.
    This can then be used to get the corresponding spectra in chunks at a
    later stage.

    Parameters
    ----------
    filename : str
        Name of the savefile
    clobber : bool
        Whether to remake the list file
    verbose : bool
        Sets print level.

    Returns
    ------

    Notes
    -----
    Saves a list of interesting SDSS id's for all galaxies with spectra
    """
    if os.path.isfile(filename):
        if clobber:
            query = ''' SELECT s.ra, s.dec, s.mjd, s.plate, s.fiberID
                        FROM  PhotoObjAll AS p JOIN SpecObjAll s ON p.objID = s.bestObjID
                        WHERE s.class='GALAXY' AND s.subclass='STARBURST'
                    '''
            res = SDSS.query_sql(query, data_release=12)
            if verbose:
                print(res)
            save_asc.write(res, filename, overwrite=True)
        else:
            if verbose:
                print('File exists. Skipping query')
    else:
        query = ''' SELECT s.ra, s.dec, s.mjd, s.plate, s.fiberID
                    FROM SpecObjAll AS s
                    WHERE s.class='GALAXY' AND s.subclass='STARBURST'
                '''
        res = SDSS.query_sql(query, data_release=12)
        if verbose:
            print(res)
        save_asc.write(res, filename, overwrite=True)


def load_table(listname, length=10, all=False):
    """
    Loads a section of the SDSS id list and removes it from the coordinate list

    Parameters
    ----------
    listname : str
        Name of the file to load from
    length : int
        Length of chunk to return
    all : bool
        If true returns the full list

    Returns
    -------
    table : pandas.DataFrame
        DataFrame with the information needed by SDSS.get_spectra.
    remaining : int
        Remaining rows in list
    """
    res = save_asc.read(listname)
    if all:
        return res

    try:
        table = res[:length]
        res = res[length:]
        save_asc.write(res, listname, overwrite=True)
        remaining = len(res)
    except IndexError:
        table = res
        remaining = 0

    return table, remaining


def chunk_table(table, chunksize=10):
    """
    Separates the table into multiprocessing chunks

    Parameters
    ----------
    table : astropy.table
        Full table object
    chunksize : int, optional
        Default size of chunks is 10

    Returns
    -------
    result : list

    """
    remaining = 1
    result = []
    while remaining > 0:
        try:
            res = copy.deepcopy(table[:chunksize])
            table = table[chunksize:]
            # save_asc.write(table, listname, overwrite=True)
            remaining = len(table)
        except IndexError:
            res = copy.deepcopy(table)
            remaining = 0
        result.append(res)
    return result


def download_spectra(table, data_dir, save_raw=True, raw_dir=None):
    """
    Downloads SDSS spectra

    Parameters
    ----------
    table : AstroPy.Table
        Table with coordinates
    data_dir : str
        Specifies directory where the data is saved
    save_raw : bool
        Specifies whether the raw spectra (i.e. sdss fits) should be saved
    raw_dir : str, optional
        Specifies save directory for the raw spectra. If raw_dir=None and
        save_raw is True it defaults to data_dir

    Returns
    -------
    spectra : list
        List of HDU objects
    filenames : list
        List of filenames to be used when saving linemeasurements
    """
    select = zip(table['mjd'], table['plate'], table['fiberID'])
    spectra = []
    filenames = []
    for mjd, plate, fiberID in select:
        # Get the spectrum from the SDSS
        spec = SDSS.get_spectra_async(mjd=mjd, plate=plate, fiberID=fiberID)

        # Load the HDU object
        fits_object = spec[0].get_fits()
        spectra.append(fits_object)

        # Save it to file
        filename = './temp/{}_{}_{}.fits'.format(mjd, plate, fiberID)
        filenames.append('{}_{}_{}'.format(mjd, plate, fiberID))
        try:
            fits_object.writeto(filename)
        except OSError:
            print('Spectrum already saved. Skipping')
    return spectra, filenames


def process_spectra(spectra, save_res=True, save_name=None):
    """
    Coordinates the measuring of lines and removal of continuum from sdss
    spectra. The steps that are performed are:

    - Unpacking the spectrum
    - Redshifting
    - Continuum subtraction
    - Line measuring

    Parameters
    ----------
    spectra : list
        List of HDU objects conforming to the SDSS standard
    save_res : bool
        Whether or not to save the individual dataframes
    save_dir : str
        Filenames where the dataframe pickles will be saved

    Returns
    -------
    objects : list
        List of dataframes with measured line fluxes
    """
    objects = []
    for i, spectrum in enumerate(spectra):
        wavelength, flux, z = unpack_spectrum(spectrum)
        nflux = normalize(wavelength, flux)
        zwave = wavelength / (1 + z)
        lines = measure_lines(wl=zwave, flux=nflux * 1e-17,
                              linelist='./data/lines.list')
        lum_dist = cosmo.luminosity_distance(z)
        lum_dist_cm = lum_dist.to(u.cm).value

        lines['flux'] = lines.flux * (4 * np.pi * lum_dist_cm**2)
        if save_res:
            assert save_name is not None
            # Pickle the result
            lines.to_pickle(save_name[i] + '.pkl')
        objects.append(lines)
    return objects


def unpack_spectrum(HDU_list):
    """
    Unpacks and extracts the relevant parts of an SDSS HDU list object.

    Parameters
    ----------
    HDU_list : astropy HDUlist object

    Returns
    -------
    wavelengths : ndarray
        Wavelength array
    flux : ndarray
        Flux array
    z : float
        Redshift of the galaxy
    """
    table_data = HDU_list[0].header
    z = HDU_list[2].data[0][63]
    wavelengths = 10 ** HDU_list[1].data['loglam']
    flux = HDU_list[1].data['flux']

    return wavelengths, flux, z


def normalize(wavelength, flux, kernel=51):
    """
    Function that normalizes the input spectrum using median filtering

    Parameters
    ----------
    wavelength : ndarray
        Array of wavelengths in Ångström
    flux : ndarray
        Array of same length as wavelength. Units should be erg/s/Å

    Returns
    -------
    normalized_flux : ndarray
        Normalized flux array
    """
    continuum = signal.medfilt(flux, kernel_size=kernel)
    normalized_flux = flux - continuum

    # plt.figure()
    # plt.plot(wavelength, normalized_flux)
    # plt.show()
    return normalized_flux


def measure_lines(wl, flux, linelist, method='narrowband', width=40):
    """
    Measures a set of emission lines to get fluxes that can be compared to
    cloudy.

    Parameters
    ----------
    wl : ndarray
        Array of wavelengths in Ångström
    flux : ndarray
        Continuum_subtracted flux. Array of same length as wavelength.
        Units should be erg/s/Å
    linelist : str
        List with line names and wavelengths. Tab separated
    method : {'narrowband', 'gaussian'}
        Selects the fitting method. Narrowband integrates the continuum
        subtracted spectrum over a small box around each line to get the line
        flux. Gaussian fits a set of gaussians to all the requested lines
    width : int
        Width of the narrowband.

    Returns
    -------
    results : pandas.DataFrame
        Dataframe with two columns: 'name', and 'flux' as measured from the
        spectrum
    """
    # Read in the lines we are interested in
    lines = pd.read_csv(linelist, delim_whitespace=True, header=None,
                        names=['name', 'wavelength'])
    # Create windows
    lower_edge = lines.wavelength - width / 2
    upper_edge = lines.wavelength + width / 2

    # Loop over the windows
    fluxes = []
    for le, ue in zip(lower_edge, upper_edge):
        # Select spectral region
        select = np.where((wl >= le) & (wl <= ue))
        # Integrate the region
        flux_sum = flux[select].sum()
        # print(flux[select])
        fluxes.append(flux_sum)

    results = pd.DataFrame()
    results['name'] = lines.name
    results['flux'] = fluxes

    return results
