import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from skimage import io, exposure, segmentation

# assign output directory
path_out = '../../output/fig2/a'
os.makedirs(path_out, exist_ok=True)

# load data
timepoints = [9, 16, 33]
df = pd.read_csv('./fig2a.csv')
df2 = df[df['TimePoint'].isin(timepoints)]

# plot time series
sb.lineplot(x='Time', y='MeanNoBgCorrectedCh00_norm', data=df, color='black')
sb.scatterplot(x='Time', y='MeanNoBgCorrectedCh00_norm', data=df2, color='black')
plt.ylim(0.00, 1.05)
plt.savefig(os.path.join(path_out, 'timeseries.pdf'), bbox_inches='tight')
plt.close()

# crop around single cells matching example time series
timepoint_pattern = re.compile(r't(\d+)_')
df = pd.read_csv('./220524DS30_p0011-002DS_QTFyDetectionCh2SegMethod0.csv')
subdirs = ['bf', 'p65', 'nuc', 'mask', 'contour', 'pred']
for subdir in subdirs:
    os.makedirs(os.path.join(path_out, subdir), exist_ok=True)

paths_bf = [
        './220524DS30_p0011/220524DS30_p0011_t00011_z001_w00.png',
        './220524DS30_p0011/220524DS30_p0011_t00016_z001_w00.png',
        './220524DS30_p0011/220524DS30_p0011_t00025_z001_w00.png',
        './220524DS30_p0011/220524DS30_p0011_t00033_z001_w00.png',
        './220524DS30_p0011/220524DS30_p0011_t00036_z001_w00.png',
        './220524DS30_p0011/220524DS30_p0011_t00044_z001_w00.png',
        './220524DS30_p0011/220524DS30_p0011_t00061_z001_w00.png',
        './220524DS30_p0011/220524DS30_p0011_t00068_z001_w00.png',
        './220524DS30_p0011/220524DS30_p0011_t00081_z001_w00.png',
        './220524DS30_p0011/220524DS30_p0011_t00091_z001_w00.png',
        './220524DS30_p0011/220524DS30_p0011_t00101_z001_w00.png',
        ]
paths_p65 = [
        './220524DS30_p0011/220524DS30_p0011_t00011_z001_w01.png',
        './220524DS30_p0011/220524DS30_p0011_t00016_z001_w01.png',
        './220524DS30_p0011/220524DS30_p0011_t00025_z001_w01.png',
        './220524DS30_p0011/220524DS30_p0011_t00033_z001_w01.png',
        './220524DS30_p0011/220524DS30_p0011_t00036_z001_w01.png',
        './220524DS30_p0011/220524DS30_p0011_t00044_z001_w01.png',
        './220524DS30_p0011/220524DS30_p0011_t00061_z001_w01.png',
        './220524DS30_p0011/220524DS30_p0011_t00068_z001_w01.png',
        './220524DS30_p0011/220524DS30_p0011_t00081_z001_w01.png',
        './220524DS30_p0011/220524DS30_p0011_t00091_z001_w01.png',
        './220524DS30_p0011/220524DS30_p0011_t00101_z001_w01.png',
        ]
paths_nuc = [
        './220524DS30_p0011/220524DS30_p0011_t00011_z001_w02.png',
        './220524DS30_p0011/220524DS30_p0011_t00016_z001_w02.png',
        './220524DS30_p0011/220524DS30_p0011_t00025_z001_w02.png',
        './220524DS30_p0011/220524DS30_p0011_t00033_z001_w02.png',
        './220524DS30_p0011/220524DS30_p0011_t00036_z001_w02.png',
        './220524DS30_p0011/220524DS30_p0011_t00044_z001_w02.png',
        './220524DS30_p0011/220524DS30_p0011_t00061_z001_w02.png',
        './220524DS30_p0011/220524DS30_p0011_t00068_z001_w02.png',
        './220524DS30_p0011/220524DS30_p0011_t00081_z001_w02.png',
        './220524DS30_p0011/220524DS30_p0011_t00091_z001_w02.png',
        './220524DS30_p0011/220524DS30_p0011_t00101_z001_w02.png',
        ]
paths_mask = [
        './220524DS30_p0011/220524DS30_p0011_t00011_z001_w02_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00016_z001_w02_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00025_z001_w02_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00033_z001_w02_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00036_z001_w02_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00044_z001_w02_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00061_z001_w02_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00068_z001_w02_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00081_z001_w02_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00091_z001_w02_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00101_z001_w02_m00_mask.png',
        ]
paths_pred = [
        './220524DS30_p0011/220524DS30_p0011_t00011_z001_w00_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00016_z001_w00_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00025_z001_w00_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00033_z001_w00_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00036_z001_w00_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00044_z001_w00_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00061_z001_w00_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00068_z001_w00_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00081_z001_w00_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00091_z001_w00_m00_mask.png',
        './220524DS30_p0011/220524DS30_p0011_t00101_z001_w00_m00_mask.png',
        ]

paths_bf.sort()
paths_p65.sort()
paths_nuc.sort()
paths_mask.sort()
paths_pred.sort()

bp_wp_bf = {
        '220524DS30_p0011_t00011_z001_w00.png': [100, 180],
        '220524DS30_p0011_t00016_z001_w00.png': [100, 180],
        '220524DS30_p0011_t00025_z001_w00.png': [100, 180],
        '220524DS30_p0011_t00033_z001_w00.png': [100, 180],
        '220524DS30_p0011_t00036_z001_w00.png': [100, 180],
        '220524DS30_p0011_t00044_z001_w00.png': [100, 180],
        '220524DS30_p0011_t00061_z001_w00.png': [100, 180],
        '220524DS30_p0011_t00068_z001_w00.png': [100, 180],
        '220524DS30_p0011_t00081_z001_w00.png': [100, 180],
        '220524DS30_p0011_t00091_z001_w00.png': [100, 180],
        '220524DS30_p0011_t00101_z001_w00.png': [100, 180],
        }

bp_wp_nuc = {
        '220524DS30_p0011_t00011_z001_w02.png': [20, 130],
        '220524DS30_p0011_t00016_z001_w02.png': [20, 130],
        '220524DS30_p0011_t00025_z001_w02.png': [20, 130],
        '220524DS30_p0011_t00033_z001_w02.png': [20, 130],
        '220524DS30_p0011_t00036_z001_w02.png': [20, 130],
        '220524DS30_p0011_t00044_z001_w02.png': [20, 130],
        '220524DS30_p0011_t00061_z001_w02.png': [20, 130],
        '220524DS30_p0011_t00068_z001_w02.png': [20, 130],
        '220524DS30_p0011_t00081_z001_w02.png': [20, 130],
        '220524DS30_p0011_t00091_z001_w02.png': [20, 130],
        '220524DS30_p0011_t00101_z001_w02.png': [20, 130],
        }

bp_wp_p65 = {
        '220524DS30_p0011_t00011_z001_w01.png': [40, 160],
        '220524DS30_p0011_t00016_z001_w01.png': [40, 160],
        '220524DS30_p0011_t00025_z001_w01.png': [40, 160],
        '220524DS30_p0011_t00033_z001_w01.png': [40, 160],
        '220524DS30_p0011_t00036_z001_w01.png': [40, 160],
        '220524DS30_p0011_t00044_z001_w01.png': [40, 160],
        '220524DS30_p0011_t00061_z001_w01.png': [40, 160],
        '220524DS30_p0011_t00068_z001_w01.png': [40, 160],
        '220524DS30_p0011_t00081_z001_w01.png': [40, 160],
        '220524DS30_p0011_t00091_z001_w01.png': [40, 160],
        '220524DS30_p0011_t00101_z001_w01.png': [40, 160],
        }

for path_bf, path_p65, path_nuc, path_mask, path_pred in zip(paths_bf, paths_p65, paths_nuc, paths_mask, paths_pred):
    # get timepoint
    i = int(timepoint_pattern.search(path_bf).group(1))

    # get centroid
    x = int(df.loc[df.TimePoint == i+1, 'XMorphologyCh02'].values)
    y = int(df.loc[df.TimePoint == i+1, 'YMorphologyCh02'].values)

    # load images
    bf = io.imread(path_bf)[:, :, 0]
    p65 = io.imread(path_p65)[:, :, 0]
    nuc = io.imread(path_nuc)[:, :, 0]
    mask = io.imread(path_mask)
    pred = io.imread(path_pred)

    # crop
    bf = bf[y-40:y+40, x-40:x+40] # crop out 80x80 px
    p65 = p65[y-40:y+40, x-40:x+40]
    nuc = nuc[y-40:y+40, x-40:x+40]
    mask = mask[y-40:y+40, x-40:x+40]
    pred = pred[y-40:y+40, x-40:x+40]

    # spread contrast
    bf_bp = bp_wp_bf[os.path.basename(path_bf)][0]
    bf_wp = bp_wp_bf[os.path.basename(path_bf)][1]
    p65_bp = bp_wp_p65[os.path.basename(path_p65)][0]
    p65_wp = bp_wp_p65[os.path.basename(path_p65)][1]
    nuc_bp = bp_wp_nuc[os.path.basename(path_nuc)][0]
    nuc_wp = bp_wp_nuc[os.path.basename(path_nuc)][1]

    bf = exposure.rescale_intensity(bf, in_range=(bf_bp, bf_wp), out_range=(0, 255))
    p65 = exposure.rescale_intensity(p65, in_range=(p65_bp, p65_wp), out_range=(0, 255))
    nuc = exposure.rescale_intensity(nuc, in_range=(nuc_bp, nuc_wp), out_range=(0, 255))

    # save p65 as green image
    p65 = np.stack([np.zeros_like(p65), p65, np.zeros_like(p65)], axis=-1).astype('uint8')

    contour = segmentation.mark_boundaries(p65, mask, color=(1, 1, 1), mode='thick')
    contour = (contour * 255).astype(np.uint8)

    # save
    io.imsave(os.path.join(path_out, 'bf', f'bf_{i}.png'), bf.astype('uint8'))
    io.imsave(os.path.join(path_out, 'p65', f'p65_{i}.png'), p65)
    io.imsave(os.path.join(path_out, 'contour', f'contour_{i}.png'), contour)
    io.imsave(os.path.join(path_out, 'nuc', f'nuc_{i}.png'), nuc.astype('uint8'))
    io.imsave(os.path.join(path_out, 'mask', f'mask_{i}.png'), mask.astype('uint8'))
    io.imsave(os.path.join(path_out, 'pred', f'pred_{i}.png'), pred.astype('uint8'))
