import numpy as np
import scipy as sp
import pandas as pd
import pyCloudy as pc
import matplotlib.pyplot as plt

# =============================================================================
#                            CODE DESCRIPTION
# =============================================================================
"""
 The code will contain routines for setting up cloudy model grids and functions
 for running and saving the routines.

 Functions:
    createGrid      - Sets up the parameter grid for which the cloudy models
                        will be run
    writeResult     - Write the Cloudy outputs to file
    runCloudy       - Runs Cloudy for a given node in the parameter grid
    distributeRuns  - Function that spreads cloudy runs over n-1 cores.
    testCloudy      - Function that runs the example from:
"""


# =============================================================================
#                  RUN TEST SETUP OF CLOUDY MODELS
# =============================================================================
def testCloudy():
    dir_ = './cloudy_cache/'
    # Define some parameters of the model:
    model_name = 'model_1'
    full_model_name = '{0}{1}'.format(dir_, model_name)
    dens = 2.       # log cm-3
    Teff = 45000.   # K
    qH = 47.        # s-1
    r_min = 5e17    # cm
    dist = 1.26     # kpc

    # these are the commands common to all the models (here only one ...)
    options = ('no molecules',
               'no level2 lines',
               'no fine opacities',
               'atom h-like levels small',
               'atom he-like levels small',
               'COSMIC RAY BACKGROUND',
               'element limit off -8',
               'print line optical depth',
               )

    emis_tab_c13 = ['H  1  4861',
                    'H  1  6563',
                    'He 1  5876',
                    'N  2  6584',
                    'O  1  6300',
                    'O II  3726',
                    'O II  3729',
                    'O  3  5007',
                    'TOTL  4363',
                    'S II  6716',
                    'S II 6731',
                    'Cl 3 5518',
                    'Cl 3 5538',
                    'O  1 63.17m',
                    'O  1 145.5m',
                    'C  2 157.6m']
    emis_tab = ['H  1  4861.36A',
                'H  1  6562.85A',
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
                'Cl 3  5537.87A',
                'O  1  63.1679m',
                'O  1  145.495m',
                'C  2  157.636m']
    abund = {'He': -0.92, 'C': 6.85 - 12, 'N': -4.0, 'O': -3.40, 'Ne': -4.00,
             'S': -5.35, 'Ar': -5.80, 'Fe': -7.4, 'Cl': -7.00}

    # Defining the object that will manage the input file for Cloudy
    c_input = pc.CloudyInput(full_model_name)

    # Filling the object with the parameters
    # Defining the ionizing SED: Effective temperature and luminosity.
    # The lumi_unit is one of the Cloudy options, like "luminosity solar",
    # "q(H)", "ionization parameter", etc...
    c_input.set_BB(Teff=Teff, lumi_unit='q(H)', lumi_value=qH)

    # Defining the density. You may also use set_dlaw(parameters) if you have
    # a density law defined in dense_fabden.cpp.
    c_input.set_cste_density(dens)

    # Defining the inner radius. A second parameter would be the outer radius
    # (matter-bounded nebula).
    c_input.set_radius(r_in=np.log10(r_min))
    c_input.set_abund(ab_dict=abund, nograins=True)
    c_input.set_other(options)

    # (0) for no iteration, () for one iteration, (N) for N iterations:
    c_input.set_iterate()
    # () or (True) : sphere, or (False): open geometry:
    c_input.set_sphere()
    # better use read_emis_file(file) for long list of lines, where file is an
    # external file:
    c_input.set_emis_tab(emis_tab)
    # unit can be 'kpc', 'Mpc', 'parsecs', 'cm'. If linear=False
    # the distance is in log:
    c_input.set_distance(dist=dist, unit='kpc', linear=True)

    # Writing the Cloudy inputs. to_file for writing to a file
    # (named by full_model_name). verbose to print on the screen.
    c_input.print_input(to_file=True, verbose=False)

    # Tell pyCloudy where your cloudy executable is:
    pc.config.cloudy_exe = '~/Documents/Cloudy/source/cloudy.exe'

    # Running Cloudy with a timer. Here we reset it to 0.
    pc.log_.timer('Starting Cloudy', quiet=True, calling='test1')
    c_input.run_cloudy()
    pc.log_.timer('Cloudy ended after seconds:', calling='test1')

    # Reading the Cloudy outputs in the Mod CloudyModel object
    Mod = pc.CloudyModel(full_model_name)

    plt.figure(figsize=(10, 10))
    plt.semilogy(Mod.get_cont_x(unit='Ang'),
                 Mod.get_cont_y(cont='incid', unit='esAc'), label='Incident')
    plt.semilogy(Mod.get_cont_x(unit='Ang'),
                 Mod.get_cont_y(cont='diffout', unit='esAc'), label='Diff Out')
    plt.semilogy(Mod.get_cont_x(unit='Ang'),
                 Mod.get_cont_y(cont='ntrans', unit='esAc'), label='Net Trans')
    plt.xlim((3800, 9200))
    # plt.ylim((1e-9, 1e1))
    plt.xlabel('Angstrom')
    plt.ylabel('Erg/s/Å/cm2')
    plt.legend(loc=4)
    plt.show()
