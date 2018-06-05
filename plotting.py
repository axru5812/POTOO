"""
Script that runs the classification.
"""
import pandas as pd
import numpy as np
import ClassificationFunctions as c
import TrainingFunctions as t
import matplotlib.pyplot as plt
from PythonUtilities.graphics import plotconfig

data = c.load_data(standardize=False)
ha_label = data.columns[1]
hb_label = data.columns[0]
ha_hb = data[ha_label] / data[hb_label]
prob_caseA = len(data.index.values[np.where(ha_hb < 2.86)])
print('Number of galaxies with Ha/Hb less than 2.86: ', prob_caseA)
plt.figure()
plt.hist(ha_hb.values[np.where((ha_hb > 0) & (ha_hb < 200))], bins=1000)
print(len(ha_hb.values[np.where((ha_hb < 0))]))
print(len(ha_hb.values[np.where((ha_hb > 0) & (ha_hb < 2.86))]))
plt.xlim(0, 20)
plt.ylabel('Number')
plt.xlabel(r'H$\alpha$ / H$\beta$')
plt.axvline(2.86, c='#ef5f00')

# Cloudy Ha/Hb ratios
cloudy_data, cloudy_classes = t.load_data('./data/training_data.pkl',
                                          './data/training_class.pkl', standardize=False)
cloudyA = cloudy_data.loc[np.where(cloudy_classes['class'] == 'A')]
cloudyB = cloudy_data.loc[np.where(cloudy_classes['class'] == 'B')]
cha_label = cloudy_data.columns[1]
chb_label = cloudy_data.columns[0]
cHabA = cloudyA[cha_label] / cloudyA[chb_label]
cHabB = cloudyB[cha_label] / cloudyB[chb_label]

plt.figure()
plt.hist(cHabA, bins=200, alpha=0.5, label='caseA')
plt.hist(cHabB, bins=200, alpha=0.5, label='caseB')
plt.legend()
plt.ylabel('Number')
plt.xlabel(r'H$\alpha$ / H$\beta$')
plt.axvline(2.86, c='#ef5f00')

plt.show()
