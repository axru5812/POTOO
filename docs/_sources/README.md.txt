![Potoo logo](../POTOO/POTOO_big_logo.png)

**Properties of line Transitions in non-Ordinary Objects**

**A project investigating the balmer line transitions looking for potential non-caseB galaxies in the SDSS**

-------------------------------------------------------------------------------

## Investigating unphysical Ha/Hb ratios

### General motivation ###

Many galaxies in the SDSS shows ratios of Ha to Hb that are below the 2.86 ratio that is predicted from the CaseB recombination scenario.

Most likely the vast majority of these low ab-ratios are caused by measurement errors but there is a possibility that some of the measurements are actually reflecting the true physical ratios in the galaxies.

The main point of this project is to try to establish a sample of galaxies that have true HaHb-ratios below 2.86 using a machine learning approach to deal with the large data volume.

### Project outline ###
c.f. GAME Paper (Ucci+18)

1. Create a library of spectra that fall in case A vs case B classes with varying other parameters.
2. Train a classification algorithm on the two labels
3. Use cross validation to characterize model performance
4. Run the classification algorithm on all SDSS emission line galaxy spectra. This will probably yield a reasonably good preselection of galaxies that are likely to have anomalous line ratios
5. Test the Balmer line shapes for gaussianity to remove any that have high probability of being anomalous due to faulty observations
6. Also a signal to noise cut should probably be applied
7. Calculate the implied optical depths, region sizes and other interesting physical parameters of the galaxy.

### Outstanding questions ###
1. How to deal with reddening in the training spectra
2. Do we need only balmer lines or full cloudy models for the computation
3. How do we define case A vs B in terms of the input parameters of cloudy
4. What ML algorithm is suitable for this classification
5. What method should be used for defining the Gaussianity of the Ha line
6. Potentially include bootstrapping for determinations of uncertainties in predictions and mitigate noise effects. GAME paper suggests that including noise in the training set is also advantageous.
