import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

# define output directory
path_out = '../output/suppl_fig1'
os.makedirs(path_out, exist_ok=True)

# load data
df = pd.read_csv('./suppl_fig1.csv')

# plot histogram
cm = 1/2.54
fig, ax = plt.subplots(figsize=(8.5*cm, 3.24*cm)) # width, height
fig = sb.swarmplot(x='lr', y='f1', data=df, color='#000000', marker='o', size=6, dodge=True)
plt.ylim(0.5, 1.05)
plt.savefig(os.path.join(path_out, 'suppl_fig1.pdf'), bbox_inches='tight')

plt.close()
