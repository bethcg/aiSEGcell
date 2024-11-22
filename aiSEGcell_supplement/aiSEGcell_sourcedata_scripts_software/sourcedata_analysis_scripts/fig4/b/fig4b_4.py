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

track_numbers = [1, 2, 3]

# plot horizontal stacked bar chart
cm = 1/2.54
colors = ['#000000', '#7F7F7F', '#FFFFFF']
width = 0.35
bottom = np.array([0, 0, 0])
fig, ax = plt.subplots(figsize=(2.24*cm, 3.76*cm))

for i, cc in enumerate(['G1', 'S', 'G2']):
    counts = (df.loc[df.TrackNumber == 1, 'cell_cycle'] == cc).sum()
    counts_plot = np.array([0, counts, 0])
    ax.bar([1.5, 2, 2.5], counts_plot, width, bottom=bottom, color=colors[i])
    bottom += np.repeat(counts, 3)

for i, cc in enumerate(['G1', 'S', 'G2']):
    counts_d1 = (df.loc[df.TrackNumber == 2, 'cell_cycle'] == cc).sum()
    counts_d2 = (df.loc[df.TrackNumber == 3, 'cell_cycle'] == cc).sum()
    counts = [counts_d1, 0, counts_d2]
    ax.bar([1.5, 2, 2.5], counts, width, bottom=bottom, color=colors[i])
    bottom += counts

ax.set_xticks([1.5, 2, 2.5], labels=['d1', 'm', 'd2'])
ax.set_ylim([0, 160])
plt.savefig(os.path.join(path_out, f'cc.pdf'), bbox_inches='tight')
plt.close()

