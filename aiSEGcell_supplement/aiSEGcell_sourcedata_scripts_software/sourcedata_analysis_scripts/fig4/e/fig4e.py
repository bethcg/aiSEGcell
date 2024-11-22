import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from scipy.stats import chisquare
from tqdm import tqdm
from typing import List, Tuple


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
                        default='../data_all.csv',
                        help='Path to the classification data.')

    parser.add_argument('--out',
                        type=str,
                        default='../../output/fig4/e',
                        help='Path to output directory.')

    return parser.parse_args()


def get_onset(data: pd.DataFrame) -> Tuple[List[bool], List[int]]:
    '''
    Determine onset of marker expression for each cell in tree.
    '''
    assert len(np.unique(data.TrackNumber)) == 3, "Tree must consist of 3 cells."

    onset = [0, 0, 0]
    expression = [False, False, False]

    for tn in (0, 1, 2):
        tmp = data.loc[data.TrackNumber == (tn + 1), :]
        tmp = tmp.iloc[::-1].reset_index(drop=True)

        for i in range(len(tmp)):
            onset[tn] += tmp.iloc[i, 2]
            
            if onset[tn] < (i + 1):
                break
        
        expression[tn] = onset[tn] > 5
        onset[tn] = tmp.loc[onset[tn] - 1, 'TimePoint'] if expression[tn] else 999

    return expression, onset


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

    # load data
    data = pd.read_csv(data_path, sep=',')
    data = data.loc[data.ResponseType != 'unclear', :].reset_index(drop=True)
    ids = [idx[:20] for idx in data.Identification]
    data.loc[:, 'Identification'] = ids
    movies = np.unique(data.Experiment)

    # remove outliers
    outlier = [
            '230624DS30_p0016-002',
            '230629DS30_p0016-001',
            ]

    # these 3 traces have been manually confirmed to express cd115 even though automatic detection misses it
    manual_confirmed = [
            '230629DS30_p0023-002',
            '230629DS30_p0026-001',
            '230629DS30_p0026-002',
            ]

    data = data.loc[~data.Identification.isin(outlier), :].reset_index(drop=True)

    # get onset of marker expression for ly6c/cd115
    data.loc[:, 'expression_ly6c'] = False
    data.loc[:, 'ly6c_onset'] = 999
    data.loc[:, 'expression_cd115'] = False
    data.loc[:, 'cd115_onset'] = 999

    cols = ['Identification', 'TrackNumber', 'Cytokine', 'ResponseType', 'expression_ly6c', 'onset_ly6c', 'expression_cd115', 'onset_cd115']
    onsets = pd.DataFrame(columns=cols)
    with tqdm(total=len(np.unique(data.Identification)), desc=f"Compute onset...", unit="trace") as pbar:
        for idx in np.unique(data.Identification):
            # get ly6c threshold
            mu_ly6c = np.mean(data.loc[(data.Identification == idx) & (data.TimePoint < 8), 'MeanNoBgCorrectedCh04_cell'])
            sigma_ly6c = np.std(data.loc[(data.Identification == idx) & (data.TimePoint < 8), 'MeanNoBgCorrectedCh04_cell'])
            data.loc[data.Identification == idx, 'MeanNoBgCorrectedCh04_cell_norm'] = data.loc[data.Identification == idx, 'MeanNoBgCorrectedCh04_cell'] / mu_ly6c
            data.loc[data.Identification == idx, 'tau_ly6c'] = data.loc[data.Identification == idx, 'MeanNoBgCorrectedCh04_cell_norm'] > (1 + 5 * sigma_ly6c / mu_ly6c)

            # get cd115 threshold
            mu_cd115 = np.mean(data.loc[(data.Identification == idx) & (data.TimePoint < 8), 'MeanNoBgCorrectedCh03_cell'])
            data.loc[data.Identification == idx, 'MeanNoBgCorrectedCh03_cell_norm'] = data.loc[data.Identification == idx, 'MeanNoBgCorrectedCh03_cell'] / mu_cd115
            data.loc[data.Identification == idx, 'tau_cd115'] = data.loc[data.Identification == idx, 'MeanNoBgCorrectedCh03_cell_norm'] > 3

            # compute if ly6c_sd3 is true for at least 6 consecutive timepoints until the end of TrackNumber
            expression_ly6c, onset_ly6c = get_onset(data.loc[data.Identification == idx, ['TrackNumber', 'TimePoint', 'tau_ly6c']])
            expression_cd115, onset_cd115 = get_onset(data.loc[data.Identification == idx, ['TrackNumber', 'TimePoint', 'tau_cd115']])

            tmp = pd.DataFrame({
                    'Identification': np.repeat(idx, 3),
                    'TrackNumber': [1, 2, 3],
                    'Cytokine': np.repeat(data.loc[data.Identification == idx, 'Cytokine'].iloc[0], 3),
                    'ResponseType': np.repeat(data.loc[data.Identification == idx, 'ResponseType'].iloc[0], 3),
                    'expression_ly6c': expression_ly6c,
                    'onset_ly6c': onset_ly6c,
                    'expression_cd115': [False, True, True] if idx in manual_confirmed else expression_cd115,
                    'onset_cd115': onset_cd115,
                })
            onsets = pd.concat([onsets, tmp], axis=0, ignore_index=True)
            data.loc[data.Identification == idx, 'expression_ly6c'] = any(expression_ly6c)
            data.loc[data.Identification == idx, 'onset_ly6c'] = min(onset_ly6c)
            data.loc[data.Identification == idx, 'expression_cd115'] = True if idx in manual_confirmed else any(expression_cd115)
            data.loc[data.Identification == idx, 'onset_cd115'] = min(onset_cd115)

            pbar.update()

    onsets.to_csv(os.path.join(out_path, 'onsets.csv'), index=False)

    # add differentiation categories based ly6c/cd115 expression
    data = data.loc[data.TimePoint == 1, :].reset_index(drop=True)
    data.loc[:, 'differentiation'] = 1

    for idx in data.Identification:
        if (data.loc[data.Identification == idx, 'expression_ly6c'].values[0] == 0) & (data.loc[data.Identification == idx, 'expression_cd115'].values[0] == 0):
            data.loc[data.Identification == idx, 'differentiation'] = 1
        elif (data.loc[data.Identification == idx, 'expression_ly6c'].values[0] == 1) & (data.loc[data.Identification == idx, 'expression_cd115'].values[0] == 0):
            data.loc[data.Identification == idx, 'differentiation'] = 2
        elif (data.loc[data.Identification == idx, 'expression_ly6c'].values[0] == 0) & (data.loc[data.Identification == idx, 'expression_cd115'].values[0] == 1):
            data.loc[data.Identification == idx, 'differentiation'] = 3
        else:
            data.loc[data.Identification == idx, 'differentiation'] = 4

    data.to_csv(os.path.join(out_path, 'differentiation.csv'), index=False)

    # compute statistics
    statistics = pd.DataFrame(columns=['Test', 'obs', 'f_obs', 'exp', 'f_exp', 'Statistic', 'p-value'])
    _, f_obs = np.unique(data.loc[data.ResponseType == 'oscillatory', 'differentiation'], return_counts=True)
    f_obs = np.array([f_obs[0], 0, f_obs[1], f_obs[2]])
    _, f_exp = np.unique(data.loc[data.ResponseType == 'non-responsive', 'differentiation'], return_counts=True)
    f_exp_adj = f_exp / np.sum(f_exp) * np.sum(f_obs)
    chi2, p_val = chisquare(f_obs, f_exp_adj)

    statistics_tmp = pd.DataFrame({
            'Test': ['chisquare'],
            'obs': ['oscillatory'],
            'f_obs': [f_obs],
            'exp': ['non-responsive'],
            'f_exp': [f_exp],
            'Statistic': [chi2],
            'p-value': [p_val]
        })
    statistics = pd.concat([statistics, statistics_tmp], axis=0, ignore_index=True)

    _, f_obs = np.unique(data.loc[data.ResponseType == 'transient', 'differentiation'], return_counts=True)
    f_obs = np.array([f_obs[0], 0, f_obs[1], f_obs[2]])
    _, f_exp = np.unique(data.loc[data.ResponseType == 'non-responsive', 'differentiation'], return_counts=True)
    f_exp_adj = f_exp / np.sum(f_exp) * np.sum(f_obs)
    chi2, p_val = chisquare(f_obs, f_exp_adj)

    statistics_tmp = pd.DataFrame({
            'Test': ['chisquare'],
            'obs': ['transient'],
            'f_obs': [f_obs],
            'exp': ['non-responsive'],
            'f_exp': [f_exp],
            'Statistic': [chi2],
            'p-value': [p_val]
        })
    statistics = pd.concat([statistics, statistics_tmp], axis=0, ignore_index=True)

    _, f_obs = np.unique(data.loc[data.ResponseType == 'oscillatory', 'differentiation'], return_counts=True)
    _, f_exp = np.unique(data.loc[data.ResponseType == 'transient', 'differentiation'], return_counts=True)
    f_exp_adj = f_exp / np.sum(f_exp) * np.sum(f_obs)
    chi2, p_val = chisquare(f_obs, f_exp_adj)

    statistics_tmp = pd.DataFrame({
            'Test': ['chisquare'],
            'obs': ['oscillatory'],
            'f_obs': [f_obs],
            'exp': ['transient'],
            'f_exp': [f_exp],
            'Statistic': [chi2],
            'p-value': [p_val]
        })
    statistics = pd.concat([statistics, statistics_tmp], axis=0, ignore_index=True)
    statistics.to_csv(os.path.join(out_path, 'statistics.csv'), index=False)

    # plot bar-plots of expression_ly6c by ResponseType and Cytokine
    colors = ['#66C2A5', '#FF5733', '#FFD700', '#800080']
    sb.countplot(x='ResponseType', hue='differentiation', data=data, palette=sb.color_palette(colors))
    plt.savefig(os.path.join(out_path, 'fig4e.pdf'))
    plt.close()


if __name__ == '__main__':
    main()
