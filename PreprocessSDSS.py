"""
Script that coordinates the download of data from the sdss database.
The steps performed are:

1. Construction of a list of SDSS id's
2. Multiprocessed download of the spectra 10 at a time (saved in ./temp/)
3. Remove continuum from the data (median filtering)
4. Measure the lines as 40 Å narrowband fluxes
5. Save the results as pickled dataframes in ./data/lines/

"""

import SDSSmanagement as sdss
import pandas as pd
import numpy as np
from astropy import coordinates as coords
from astropy import units as u
from astropy import table
from astropy.io import ascii as save_asc
from multiprocess import Pool
import tqdm
import os
import sys
import warnings
import shutil

def main():
    # Create SDSS list
    sdss.make_SDSS_idlist(filename='SDSS_coords.list', clobber=False)

    remaining_lines = 1
    while remaining_lines > 0:
        # Load a chunk
        tab, remaining_lines = sdss.load_table('SDSS_coords.list')
        print('Remaining spectra: ', remaining_lines)
        # Download spectra
        spectra, names = sdss.download_spectra(tab, 'results')

        # Process spectra
        namelist = ['./data/lines/' + name for name in names]
        lines = sdss.process_spectra(spectra, save_name=namelist)


def multidownload(tab):
    try:
        # Download spectra
        spectra, names = sdss.download_spectra(tab, 'results')

        # Process spectra
        namelist = ['./data/lines/' + name for name in names]
        lines = sdss.process_spectra(spectra, save_name=namelist)
    except Exception as e:
        warnings.warn('The following errors occured:\n' + str(e))
        try:
            shutil.rmtree('/home/axel/.astropy/cache/download/py3/lock')
        except FileNotFoundError:
            pass


def mute():
    sys.stdout = open(os.devnull, 'w')


if __name__ == '__main__':
    print('Making ID list')
    sdss.make_SDSS_idlist(filename='SDSS_coords.list', clobber=True,
                          verbose=False)
    tab = sdss.load_table('SDSS_coords.list', all=True)
    print('Identified {} galaxies'.format(len(tab)))

    print('Chunking the input')
    chunks = sdss.chunk_table(tab)

    print('Starting the processing')
    pool = Pool(processes=8, initializer=mute)
    for _ in tqdm.tqdm(pool.imap(multidownload, chunks), total=len(chunks)):
        pass
    # main()
