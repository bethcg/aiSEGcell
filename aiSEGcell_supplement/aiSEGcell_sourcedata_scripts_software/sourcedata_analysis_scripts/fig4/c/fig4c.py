import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from scipy.stats import chisquare


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
    desc = "Program to plot pie charts."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--data',
                        type=str,
                        default='../data_all.csv',
                        help='Path to the classification data.')

    parser.add_argument('--data_tk',
                        type=str,
                        default='./TK_blood_GMP_classifications.csv',
                        help='Path to the classification data published in Blood 2022.')

    parser.add_argument('--out',
                        type=str,
                        default='../../output/fig4/c',
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
    data_tk_path = args.data_tk
    out_path = args.out

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # load data
    data = pd.read_csv(data_path, sep=',')
    data_tk = pd.read_csv(data_tk_path, sep=',')
    ids = [idx[:20] for idx in data.Identification]
    data.loc[:, 'Identification'] = ids

    # create data_sub for ResponseType pie charts
    data_sub = data.loc[data.TimePoint == 1, :]
    data_sub = data_sub.sort_values(by = ['Identification']).reset_index(drop=True)

    # filter TK data for GMP-GMs
    data_tk = data_tk.loc[data_tk.CellType == 'GMP-', :]

    # kick out outlier traces
    data_sub = data_sub.loc[data_sub.ResponseType != 'unclear', :].reset_index(drop=True)

    # compute chisquare statistics for TNF and blank
    statistics = pd.DataFrame(columns=['Test', 'Cytokine', 'new', 'TK', 'Statistic', 'p-value'])
    _, f_obs = np.unique(data_sub.loc[data_sub.Cytokine == 'TNF 40ng/uL', 'ResponseType'], return_counts=True)
    f_obs = np.array([f_obs[0], f_obs[1], 0, f_obs[2]])
    _, f_exp = np.unique(data_tk.loc[data_tk.Cytokine == 'TNF', 'ResponseType'], return_counts=True)
    f_exp_adj = f_exp / np.sum(f_exp) * np.sum(f_obs)
    chi2, p_val = chisquare(f_obs, f_exp_adj)

    statistics_tmp = pd.DataFrame({
            'Test': ['chisquare'],
            'Cytokine': ['TNF'],
            'new': [f_obs],
            'TK': [f_exp],
            'Statistic': [chi2],
            'p-value': [p_val]
        })
    statistics = pd.concat([statistics, statistics_tmp], axis=0, ignore_index=True)

    _, f_obs = np.unique(data_sub.loc[data_sub.Cytokine == 'blank', 'ResponseType'], return_counts=True)
    f_obs = np.array([f_obs[0], 0, f_obs[1]])
    _, f_exp = np.unique(data_tk.loc[data_tk.Cytokine == 'blank', 'ResponseType'], return_counts=True)
    f_exp_adj = f_exp / np.sum(f_exp) * np.sum(f_obs)

    chi2, p_val = chisquare(f_obs, f_exp_adj)

    statistics_tmp = pd.DataFrame({
            'Test': ['chisquare'],
            'Cytokine': ['blank'],
            'new': [f_obs],
            'TK': [f_exp],
            'Statistic': [chi2],
            'p-value': [p_val]
        })
    statistics = pd.concat([statistics, statistics_tmp], axis=0, ignore_index=True)

    statistics.to_csv(os.path.join(out_path, f'statistics.csv'), index=False)

    # plot bar graphs for TNF/blank and TK/our data
    colors = ['#808000', '#328A8E', '#006400', '#004080']
    ids = ['Non', 'Osc', 'Sus', 'Tra']
    _, counts = np.unique(data_sub.loc[data_sub.Cytokine == 'TNF 40ng/uL', 'ResponseType'], return_counts=True)
    counts = np.array([counts[0], counts[1], 0, counts[2]]) # add 0-counts for sustained
    percentages = counts / np.sum(counts) * 100
    fig, ax = plt.subplots()
    ax.pie(counts, labels=ids, colors=colors, startangle=90)
    plt.savefig(os.path.join(out_path, f'new_TNF_pie.pdf'))
    plt.close()

    _, counts = np.unique(data_sub.loc[data_sub.Cytokine == 'blank', 'ResponseType'], return_counts=True)
    counts = np.array([counts[0], 0, 0, counts[1]]) # add 0-counts for sustained and oscillatory
    percentages = counts / np.sum(counts) * 100
    fig, ax = plt.subplots()
    ax.pie(counts, labels=ids, colors=colors, startangle=90)
    plt.savefig(os.path.join(out_path, f'new_blank_pie.pdf'))
    plt.close()

    _, counts = np.unique(data_tk.loc[data_tk.Cytokine == 'TNF', 'ResponseType'], return_counts=True)
    percentages = counts / np.sum(counts) * 100
    fig, ax = plt.subplots()
    ax.pie(counts, labels=ids, colors=colors, startangle=90)
    plt.savefig(os.path.join(out_path, f'TK_TNF_pie.pdf'))
    plt.close()

    _, counts = np.unique(data_tk.loc[data_tk.Cytokine == 'blank', 'ResponseType'], return_counts=True)
    counts = np.array([counts[0], 0, counts[1], counts[2]]) # add 0-counts for oscillatory
    percentages = counts / np.sum(counts) * 100
    fig, ax = plt.subplots()
    ax.pie(counts, labels=ids, colors=colors, startangle=90)
    plt.savefig(os.path.join(out_path, f'TK_blank_pie.pdf'))
    plt.close()


if __name__ == '__main__':
    main()
