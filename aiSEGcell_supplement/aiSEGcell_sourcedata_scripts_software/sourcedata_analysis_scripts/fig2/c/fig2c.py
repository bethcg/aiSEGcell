import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.metrics import confusion_matrix


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
                        default='../../output/fig2/c',
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

    # load data
    data = pd.read_csv(data_path, sep=',')
    data = data.sort_values(by=['Set']).reset_index(drop=True)
    ids = [idx[:20] for idx in data.Identification]
    data.loc[:, 'Identification'] = ids
    movies = np.unique(data.Experiment)

    # create data_sub for ResponseType confusion matrix
    data_sub = data.loc[data.TimePoint == 1, :]
    data_sub = data_sub.sort_values(by = ['Identification']).reset_index(drop=True)

    # kick out outlier traces
    n_outliers_gt = len(data_sub.loc[(data_sub.Set == 'gt') & (data_sub.ResponseType == 'outlier'), :])
    n_outliers_pred = len(data_sub.loc[(data_sub.Set == 'pred') & (data_sub.ResponseType == 'outlier'), :])
    data_sub = data_sub.loc[data_sub.ResponseType != 'outlier', :].reset_index(drop=True)

    # filter traces that only occur once
    ids, counts = np.unique(data_sub.Identification, return_counts=True)
    ids_drop = ids[np.where(counts != 2)]
    data_sub = data_sub.loc[~data_sub.Identification.isin(ids_drop), :].reset_index(drop=True)

    ids, counts = np.unique(data_sub.Identification, return_counts=True)
    assert all(counts == 2), 'some traces have no matching pair'

    # plot heatmap of confusion matrix for single and all movies
    rts = np.unique(data_sub.ResponseType)

    # all movies
    conf_mat = confusion_matrix(data_sub.loc[data_sub.Set == 'gt', 'ResponseType'], data_sub.loc[data_sub.Set == 'pred', 'ResponseType'], labels = rts)
    row_sums = np.sum(conf_mat, axis=1).reshape(-1, 1)
    row_sums = np.tile(row_sums, (1, len(rts)))
    conf_mat = pd.DataFrame(data=conf_mat, index=rts, columns=rts)
    conf_mat.to_csv(os.path.join(out_path, 'all_confmat.csv'), sep=',')
    conf_mat_n = conf_mat / (row_sums + 1e-9)

    fig = sb.heatmap(conf_mat_n, cmap='Blues', vmin=0, vmax=1)
    plt.title(f'outlier_gt={n_outliers_gt}, outlier_pred={n_outliers_pred}')
    plt.savefig(os.path.join(out_path, f'all_confmat.pdf'), dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
