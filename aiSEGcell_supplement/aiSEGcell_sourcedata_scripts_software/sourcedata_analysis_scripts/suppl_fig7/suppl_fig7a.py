import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

# define output directory
path_out = '../output/suppl_fig7'
os.makedirs(path_out, exist_ok=True)

# load data
df = pd.read_csv('./suppl_fig7a.csv')

df.loc[:, 'loss_weight'] = 1

for idx in df.idx:
    if idx in ['val2', 'val17', 'val32']:
        df.loc[df.idx == idx, 'loss_weight'] = 2
    elif idx in ['val3', 'val18', 'val33']:
        df.loc[df.idx == idx, 'loss_weight'] = 3
    elif idx in ['val4', 'val19', 'val34']:
        df.loc[df.idx == idx, 'loss_weight'] = 4
    elif idx in ['val5', 'val20', 'val35']:
        df.loc[df.idx == idx, 'loss_weight'] = 5
    elif idx in ['val6', 'val21', 'val36']:
        df.loc[df.idx == idx, 'loss_weight'] = 6
    elif idx in ['val7', 'val22', 'val37']:
        df.loc[df.idx == idx, 'loss_weight'] = 7
    elif idx in ['val8', 'val23', 'val38']:
        df.loc[df.idx == idx, 'loss_weight'] = 8
    elif idx in ['val9', 'val24', 'val39']:
        df.loc[df.idx == idx, 'loss_weight'] = 9
    elif idx in ['val10', 'val25', 'val40']:
        df.loc[df.idx == idx, 'loss_weight'] = 10
    elif idx in ['val11', 'val26', 'val41']:
        df.loc[df.idx == idx, 'loss_weight'] = 20
    elif idx in ['val12', 'val27', 'val42']:
        df.loc[df.idx == idx, 'loss_weight'] = 50
    elif idx in ['val13', 'val28', 'val43']:
        df.loc[df.idx == idx, 'loss_weight'] = 100
    elif idx in ['val14', 'val29', 'val44']:
        df.loc[df.idx == idx, 'loss_weight'] = 150
    elif idx in ['val15', 'val30', 'val45']:
        df.loc[df.idx == idx, 'loss_weight'] = 200

# plot histogram
fig = sb.swarmplot(x='loss_weight', y='iou_epoch', data=df, color='#000000', marker='o', size=7, dodge=True)
plt.ylim(0, 1.05)
plt.savefig(os.path.join(path_out, 'suppl_fig7a.pdf'), bbox_inches='tight')

plt.close()
