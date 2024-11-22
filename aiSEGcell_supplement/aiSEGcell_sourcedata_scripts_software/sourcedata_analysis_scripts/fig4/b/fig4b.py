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

# plot time series
cm = 1/2.54
colors = ['#000000']
fig, ax = plt.subplots(figsize=(3.76*cm, 2.24*cm))
# sb.lineplot(x='Time', y='relSignal', data=df.loc[df.TrackNumber == 1, :], color=colors[0])
sb.lineplot(x='Time', y='relSignal', data=df, hue='TrackNumber', color=colors[0])
#plt.ylim(0.00, 0.95)
plt.xlim(0, 24)
plt.savefig(os.path.join(path_out, f'p65.pdf'), bbox_inches='tight')
plt.close()

