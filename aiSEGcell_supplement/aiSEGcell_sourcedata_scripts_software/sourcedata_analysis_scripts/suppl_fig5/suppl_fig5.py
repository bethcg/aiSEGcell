import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from scipy.stats import spearmanr, pearsonr
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

    parser.add_argument('--gt',
                        type=str,
                        default='./gt.csv',
                        help='Path to faster-based time series.')

    parser.add_argument('--pred',
                        type=str,
                        default='./pred.csv',
                        help='Path to unet-based time series.')

    parser.add_argument('--labels',
                        type=str,
                        default='./labels.csv',
                        help='Path to the classification labels.')

    parser.add_argument('--out',
                        type=str,
                        default='../output/suppl_fig5',
                        help='Path to output directory.')

    return parser.parse_args()


def main():
    '''
        Coordinates the correlation anlysis for multiple movies.
    '''
    args = args_parse()
    path_gt = args.gt
    path_pred = args.pred
    path_labels = args.labels
    path_out = args.out
    subdirs = ['mask', 'pred']

    for subdir in subdirs:
        os.makedirs(os.path.join(path_out, subdir), exist_ok=True)

    # load data
    gt = pd.read_csv(path_gt, sep=',')
    pred = pd.read_csv(path_pred, sep=',')
    labels = pd.read_csv(path_labels, sep=',')

    # filter out transient traces in GT
    labels = labels.loc[labels.label == 'transient', :].reset_index(drop=True)
    gt = gt.loc[gt.Id.isin(labels.Id), :].reset_index(drop=True)
    pred = pred.loc[pred.Id.isin(labels.Id), :].reset_index(drop=True)

    gt_sub = gt.loc[:, ('Id', 'ExperimentID', 'Timepoint', 'log2_na')]
    gt_sub = gt_sub.rename({'log2_na': 'log2'}, axis=1)
    gt_sub.loc[:, 'Set'] = 'gt'
    gt_sub = gt_sub.loc[gt_sub.Id.isin(labels.Id), :].reset_index(drop=True)
    pred_sub = pred.loc[:, ('Id', 'ExperimentID', 'Timepoint', 'log2')]
    pred_sub.loc[:, 'Set'] = 'pred'
    pred_sub = pred_sub.loc[pred_sub.Id.isin(labels.Id), :].reset_index(drop=True)
    data = pd.concat((gt_sub, pred_sub), ignore_index=True)

    # plot traces with gt and pred class labels
    timepoints = [7, 21, 35]
    data2 = data[data['Timepoint'].isin(timepoints)]
    distances = pd.DataFrame(columns=['id', 'movie', 'dist_eucl', 'pearsonr', 'spearmanr', 'lab_gt', 'lab_pred', 'match'])
    colors = ['#000000', '#28C2E5']
    signal = 'log2'
    ids = np.unique(data.Id)

    ids, counts = np.unique(data.loc[data.Timepoint == 1, 'Id'], return_counts=True)
    assert all(counts == 2), 'some traces have no matching pair'

    with tqdm(total=len(ids), desc=f"Plot traces...", unit="trace") as pbar:
        for idx in ids:
            match = 'cor'
            tmp = data.loc[data.Id == idx, :]
            tmp = tmp.sort_values(['Set', 'Timepoint']).reset_index(drop=True)
            tmp2 = data2.loc[data2.Id == idx, :]
            tmp2 = tmp2.sort_values(['Set', 'Timepoint']).reset_index(drop=True)

            lab_gt = labels.loc[labels.Id == idx, 'label'].item()
            lab_pred = labels.loc[labels.Id == idx, 'prediction'].item()
            if lab_gt != lab_pred:
                match = 'incor'

            # compute distance and correlation
            dist = np.linalg.norm(tmp.loc[tmp.Set == 'gt', signal].values - tmp.loc[tmp.Set == 'pred', signal].values)
            corr_p = pearsonr(tmp.loc[tmp.Set == 'gt', signal].values, tmp.loc[tmp.Set == 'pred', signal].values)[0]
            corr_s = spearmanr(tmp.loc[tmp.Set == 'gt', signal].values, tmp.loc[tmp.Set == 'pred', signal].values)[0]

            tmp_distances = pd.DataFrame({
                        'id': idx,
                        'movie': tmp.ExperimentID[0],
                        'dist_eucl': dist,
                        'pearsonr': corr_p,
                        'spearmanr': corr_s,
                        'lab_gt': lab_gt,
                        'lab_pred': lab_pred,
                        'match': match
                    }, index=[0])

            distances = pd.concat((distances, tmp_distances), ignore_index=True)

            plt.rcParams["figure.figsize"] = (1.68,1.525)
            plt.axes().set_yticks([-0.5,0,1])
            plt.ylim(ymin=-0.5, ymax=1.55)
            sb.lineplot(x='Timepoint', y=signal, hue='Set', data=tmp, palette=sb.color_palette(colors))
            sb.scatterplot(x='Timepoint', y=signal, data=tmp2, hue='Set', palette=sb.color_palette(colors))
            plt.title(f'gt={lab_gt}, pred={lab_pred}')
            plt.savefig(os.path.join(path_out, f'{idx}_{match}.pdf'), dpi=300, bbox_inches="tight")
            plt.close()

            pbar.update()

    distances.to_csv(os.path.join(path_out, 'distances.csv'), index=False)


if __name__ == '__main__':
    main()
