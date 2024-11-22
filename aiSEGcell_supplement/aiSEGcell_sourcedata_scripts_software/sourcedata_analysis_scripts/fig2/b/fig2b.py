import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# assign output directory
path_out = '../../output/fig2/b'
os.makedirs(path_out, exist_ok=True)

# load data
df = pd.read_csv('./fig2b.csv')
ids = np.unique(df.Identification)

# plot time series
colors = ['#000000', '#28C2E5']
for idx in ids:
    tmp = df.loc[df.Identification == idx, :].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    sb.lineplot(x='Time', y='MeanNoBgCorrectedCh00_norm', hue='Set', data=tmp, palette=sb.color_palette(colors))
    plt.ylim(0.00, 0.95)
    # plt.xlim(-1, 21)
    ax.axvline(x=1, color='r')
    plt.savefig(os.path.join(path_out, f'{idx}.pdf'), bbox_inches='tight')
    plt.close()

# save distances of individual traces
data_all = pd.read_csv('../data_all.csv')
ids = np.unique(data_all.Identification)
dist = []

for idx in ids:
    tmp = data_all.loc[data_all.Identification == idx, :].reset_index(drop=True)
    assert np.all(tmp.loc[tmp.Set == 'gt', 'TimePoint'].values == tmp.loc[tmp.Set == 'pred', 'TimePoint'].values)
    dist.append(np.linalg.norm(tmp.loc[tmp.Set == 'gt', 'MeanNoBgCorrectedCh00_norm'].values - tmp.loc[tmp.Set == 'pred', 'MeanNoBgCorrectedCh00_norm'].values))

distances = pd.DataFrame({'Identification': ids, 'eucl_dist': dist})
distances.to_csv(os.path.join(path_out, 'distances.csv'), index=False)
