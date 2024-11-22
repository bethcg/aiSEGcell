import argparse
import glob
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from skimage import io


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
    desc = "Program to plot histograms of f1 scores."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--data',
                        type=str,
                        default='./metrics_all.csv',
                        help='Path to metrics_all.csv files.')
    
    parser.add_argument('--images',
                        type=str,
                        default='./230418DS30_p0011/',
                        help='Path to image files.')

    parser.add_argument('--out',
                        type=str,
                        default='../output/suppl_fig6/',
                        help='Path to output directory.')

    return parser.parse_args()

def main():
    args = args_parse()
    path_data = args.data
    path_out = args.out
    path_images = args.images

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    df = pd.read_csv(path_data)

    features = [
            'f1_step',
            # 'loss_test_step',
            # 'tp_step',
            # 'fp_step',
            # 'fn_step',
            # 'splits_step',
            # 'merges_step',
            # 'inaccurate_masks_step',
            # 'iou_step',
            # 'iou_big_step',
            # 'iou_small_step'
            ]

    # plot different scores (x-axis) vs z (y-axis)
    for feature in features:
        cm = 1/2.54  # centimeters in inches
        plt.figure(figsize=(8.33*cm,8.30*cm))
        fig = sb.lineplot(x=feature, y='z_um', data=df, orient='y', color='black')
        if feature in ['f1_step', 'iou_step', 'iou_big_step', 'iou_small_step']:
            plt.xlim(0, 1.05)
        plt.axhline(y=-10, color='black', xmin=0, xmax=1)
        plt.axhline(y=-5, color='black', xmin=0, xmax=1)
        plt.axhline(y=-1.4, color='black', xmin=0, xmax=1)
        plt.axhline(y=0, color='black', xmin=0, xmax=1)
        plt.axhline(y=1.8, color='black', xmin=0, xmax=1)
        plt.axhline(y=5, color='black', xmin=0, xmax=1)
        plt.axhline(y=10, color='black', xmin=0, xmax=1)
        plt.yticks(np.arange(-10,12.5,2.5))
        plt.savefig(os.path.join(path_out, f'{feature}_z.pdf'), dpi=300, bbox_inches = "tight")
        plt.close()

    # generate image crops
    os.makedirs(os.path.join(path_out, "crops"), exist_ok=True)
    img_files  = glob.glob(path_images + "*.png")
    for img_file in img_files:
        img = io.imread(img_file)
        img_crop = img[752:812,1069:1129]
        io.imsave(os.path.join(path_out, 'crops', os.path.basename(img_file)), img_crop.astype('uint8'))


if __name__ == '__main__':
    main()
