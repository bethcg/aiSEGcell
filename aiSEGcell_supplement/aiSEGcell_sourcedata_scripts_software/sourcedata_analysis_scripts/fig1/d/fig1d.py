import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

# define output directory
path_out = '../../output/fig1/d'
os.makedirs(path_out, exist_ok=True)

# load data
df = pd.read_csv('./fig1d.csv')

# plot histogram
cm = 1/2.54
fig, ax = plt.subplots(figsize=(3.46*cm, 2.4*cm)) # width, height
sb.histplot(data=df, x='f1', bins=50, color='#000000', edgecolor=None)
ax.xaxis.label.set_visible(False)
ax.yaxis.label.set_visible(False)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.tight_layout()
plt.savefig(os.path.join(path_out, 'fig1_d.pdf'))

plt.close()
