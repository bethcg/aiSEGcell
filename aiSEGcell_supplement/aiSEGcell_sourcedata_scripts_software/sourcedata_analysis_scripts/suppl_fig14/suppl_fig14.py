import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from scipy.stats import chisquare
from tqdm import tqdm

def args_parse():
    '''
        Catches user input.


        Parameters
        ----------

        -


        Return
        ------

        Returns a namespace from `argparse.parse_args()`.
    '''
    desc = "Program to plot traces and image crops."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--data',
                        type=str,
                        default='./data_all.csv',
                        help='Path to the classification data.')

    parser.add_argument('--out',
                        type=str,
                        default='../output/suppl_fig14',
                        help='Path to output directory.')

    return parser.parse_args()


def main():
    '''
        Coordinates the correlation anlysis for multiple movies.
    '''
    # set random seeds
    np.random.seed(111282)

    args = args_parse()
    data_path = args.data
    out_path = args.out

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    outliers = [
            '230624DS30_p0027-001'
            ]

    # load data
    data = pd.read_csv(data_path, sep=',')
    data = data.loc[data.ResponseType != 'unclear', :].reset_index(drop=True)
    ids = [idx[:20] for idx in data.Identification]
    data.loc[:, 'Identification'] = ids
    movies = np.unique(data.Experiment)

    data = data.loc[~data.Identification.isin(outliers), :].reset_index(drop=True)
    ids = np.unique(data.Identification)

    cell_cycles = pd.DataFrame(columns=['Identification', 'Experiment', 'Cytokine', 'ResponseType', 'cc_at_stim'], dtype=float)
    with tqdm(total=len(np.unique(ids)), desc=f"Compute onset...", unit="trace") as pbar:
        for idx in np.unique(ids):
            tmp = data.loc[data.Identification == idx, :]

            df = pd.DataFrame({
                    'Identification': [idx],
                    'Experiment': [tmp.loc[:, 'Experiment'].iloc[0]],
                    'Cytokine': [tmp.loc[:, 'Cytokine'].iloc[0]],
                    'ResponseType': [tmp.loc[:, 'ResponseType'].iloc[0]],
                    'cc_at_stim': [tmp.loc[tmp.TimePoint == 7, 'cell_cycle'].item()],
                })
            cell_cycles = pd.concat((cell_cycles, df), axis=0)

            pbar.update()

    # get TNF stimulated sub-set
    cell_cycles = cell_cycles = cell_cycles.loc[cell_cycles.Cytokine == 'TNF 40ng/uL', :].reset_index(drop=True)

    cell_cycles.to_csv(os.path.join(out_path, 'cell_cycles.csv'), index=False)

    # statistic analysis
    statistics = pd.DataFrame(columns=['Test', 'Cytokine', 'n_g1', 'n_s', 'Statistic', 'p-value'])

    _, f_obs = np.unique(cell_cycles.loc[cell_cycles.cc_at_stim == 'S', 'ResponseType'], return_counts=True)
    _, f_exp = np.unique(cell_cycles.loc[cell_cycles.cc_at_stim == 'G1', 'ResponseType'], return_counts=True)
    n_g1 = np.sum(f_exp)
    n_s = np.sum(f_obs)

    f_exp_adj = f_exp / np.sum(f_exp) * np.sum(f_obs)
    chi2, p_val = chisquare(f_obs, f_exp_adj)

    statistics_tmp = pd.DataFrame({
            'Test': ['chisquare'],
            'Cytokine': ['TNF'],
            'n_g1': [n_g1],
            'n_s': [n_s],
            'Statistic': [chi2],
            'p-value': [p_val]
        })
    statistics = pd.concat([statistics, statistics_tmp], axis=0, ignore_index=True)

    statistics.to_csv(os.path.join(out_path, 'statistics.csv'), index=False)

    # plot division_tp by cytokine and response type
    colors = ['#808000', '#328A8E', '#006400', '#004080']
    ids = ['Non', 'Osc', 'Sus', 'Tra']
    _, counts = np.unique(cell_cycles.loc[(cell_cycles.cc_at_stim == 'G1'), 'ResponseType'], return_counts=True)
    counts = np.array([counts[0], counts[1], 0, counts[2]])
    fig, ax = plt.subplots()
    ax.pie(counts, labels=ids, colors=colors, startangle=90)
    plt.savefig(os.path.join(out_path, f'pie_chart_G1.pdf'))
    plt.close()

    _, counts = np.unique(cell_cycles.loc[(cell_cycles.cc_at_stim == 'S'), 'ResponseType'], return_counts=True)
    counts = np.array([counts[0], counts[1], 0, counts[2]])
    fig, ax = plt.subplots()
    ax.pie(counts, labels=ids, colors=colors, startangle=90)
    plt.savefig(os.path.join(out_path, f'pie_chart_S.pdf'))
    plt.close()


if __name__ == '__main__':
    main()
