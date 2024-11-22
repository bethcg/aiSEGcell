import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# assign output directory
path_out = '../../output/fig4/b'
os.makedirs(path_out, exist_ok=True)

# load data
df = pd.read_csv('../data_all.csv')
df = df.loc[df.Identification == '230624DS30_p0018-001DS', :].reset_index(drop=True)
mu = np.mean(df.loc[df.TimePoint < 8, 'MeanNoBgCorrectedCh04_cell'])
sigma = np.std(df.loc[df.TimePoint < 8, 'MeanNoBgCorrectedCh04_cell']) / mu
df.loc[:, 'MeanNoBgCorrectedCh04_cell_norm'] = df.loc[:, 'MeanNoBgCorrectedCh04_cell'] / mu

tau = 1 + 5 * sigma

# plot time series
cm = 1/2.54
fig, ax = plt.subplots(figsize=(3.76*cm, 2.24*cm))
sb.lineplot(x='Time', y='MeanNoBgCorrectedCh04_cell_norm', hue='TrackNumber', data=df)
plt.hlines(tau, 0, 24, colors='k', linestyles='dashed')
plt.ylim(0.5, 2.5)
plt.xlim(0, 24)
plt.savefig(os.path.join(path_out, f'ly6c.pdf'), bbox_inches='tight')
plt.close()
