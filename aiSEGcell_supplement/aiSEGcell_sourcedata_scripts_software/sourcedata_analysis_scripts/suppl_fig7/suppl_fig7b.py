import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

# define output directory
path_out = '../output/suppl_fig7'
os.makedirs(path_out, exist_ok=True)

# load data
df = pd.read_csv('./suppl_fig7b.csv')

df.loc[:, 'param_comb'] = 1

for idx in df.idx:
    if idx in ['val2', 'val14', 'val26']:
        df.loc[df.idx == idx, 'param_comb'] = 2
    elif idx in ['val3', 'val15', 'val27']:
        df.loc[df.idx == idx, 'param_comb'] = 3
    elif idx in ['val4', 'val16', 'val28']:
        df.loc[df.idx == idx, 'param_comb'] = 4
    elif idx in ['val5', 'val17', 'val29']:
        df.loc[df.idx == idx, 'param_comb'] = 5
    elif idx in ['val6', 'val18', 'val30']:
        df.loc[df.idx == idx, 'param_comb'] = 6
    elif idx in ['val7', 'val19', 'val31']:
        df.loc[df.idx == idx, 'param_comb'] = 7
    elif idx in ['val8', 'val20', 'val32']:
        df.loc[df.idx == idx, 'param_comb'] = 8
    elif idx in ['val9', 'val21', 'val33']:
        df.loc[df.idx == idx, 'param_comb'] = 9
    elif idx in ['val10', 'val22', 'val34']:
        df.loc[df.idx == idx, 'param_comb'] = 10
    elif idx in ['val11', 'val23', 'val35']:
        df.loc[df.idx == idx, 'param_comb'] = 11
    elif idx in ['val12', 'val24', 'val36']:
        df.loc[df.idx == idx, 'param_comb'] = 12

# plot histogram
fig = sb.swarmplot(x='param_comb', y='f1_epoch', data=df, color='#000000', marker='o', size=7, dodge=True)
plt.ylim(0, 1.05)
plt.savefig(os.path.join(path_out, 'suppl_fig7b.pdf'), bbox_inches='tight')

plt.close()
