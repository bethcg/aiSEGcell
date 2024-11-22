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
mu = np.mean(df.loc[df.TimePoint < 8, 'MeanNoBgCorrectedCh03_cell'])
df.loc[:, 'MeanNoBgCorrectedCh03_cell_norm'] = df.loc[:, 'MeanNoBgCorrectedCh03_cell'] / mu

tau = 3

# plot time series
cm = 1/2.54
fig, ax = plt.subplots(figsize=(3.76*cm, 2.24*cm))
sb.lineplot(x='Time', y='MeanNoBgCorrectedCh03_cell_norm', hue='TrackNumber', data=df)
plt.hlines(tau, 0, 24, colors='k', linestyles='dashed')
plt.ylim(-10, 50)
plt.xlim(0, 24)
plt.savefig(os.path.join(path_out, f'cd115.pdf'), bbox_inches='tight')
plt.close()
