import os
import numpy as np
import pandas as pd


# define output directory
path_out = '../output/suppl_tab2/'
os.makedirs(path_out, exist_ok=True)

# UNET

models = ['unet', 'SD_fluo', 'SD_he', 'cp_nuclei', 'cp_yeast_BF']

for model in models:
    print(f'\n### Processing {model}')
    # load data
    df = pd.read_csv(f'./raw_data/suppl_tab2_{model}.csv')

    # save means + stds
    thetas = np.unique(df.threshold)
    cols = ['val'] + [f'{theta:.2f}' for theta in thetas]

    mu = pd.DataFrame(columns=cols)
    sd = pd.DataFrame(columns=cols)

    # iteratively fill up mu and sd by val
    for i, val in enumerate(np.unique(df.val)):
        tmp = df[df.val == val].reset_index(drop=True)
        tmp_agg = tmp.groupby('threshold').f1.agg(['mean', 'std']).reset_index()

        # fill up mu
        mu.loc[i, 'val'] = val
        mu.iloc[i, 1:(1 + len(thetas))] = tmp_agg['mean'].values

        # fill up sd
        sd.loc[i, 'val'] = val
        sd.iloc[i, 1:(1 + len(thetas))] = tmp_agg['std'].values

    # save to csv
    mu.to_csv(os.path.join(path_out, f'{model}_mean.csv'), index=False)
    sd.to_csv(os.path.join(path_out, f'{model}_std.csv'), index=False)
