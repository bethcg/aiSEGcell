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
    desc = "Program to plot confusion matrix."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--data',
                        type=str,
                        default='./190905WW12_classification.csv',
                        help='Path to the classification data.')

    parser.add_argument('--out',
                        type=str,
                        default='../../output/fig3/c',
                        help='Path to output directory.')

    return parser.parse_args()


def main():
    '''
        Coordinates the correlation anlysis for multiple movies.
    '''
    args = args_parse()
    data_path = args.data
    out_path = args.out

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # load data
    data = pd.read_csv(data_path, sep=',')

    # plot heatmap of confusion matrix for single and all movies
    rts = ['non-responsive', 'transient', 'intermediate', 'sustained']

    # all movies
    conf_mat = confusion_matrix(data.label, data.prediction, labels = rts)
    row_sums = np.sum(conf_mat, axis=1).reshape(-1, 1)
    row_sums = np.tile(row_sums, (1, len(rts)))
    conf_mat = pd.DataFrame(data=conf_mat, index=rts, columns=rts)
    conf_mat.to_csv(os.path.join(out_path, 'all_confmat.csv'), sep=',')
    conf_mat_n = conf_mat / (row_sums + 1e-9)

    acc = np.sum(np.diag(conf_mat)) / np.sum(conf_mat.values)
    fig = sb.heatmap(conf_mat_n, cmap='Blues', vmin=0, vmax=1)
    plt.title(f'acc = {acc:.3f}')
    plt.savefig(os.path.join(out_path, f'all_confmat.pdf'), dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
