import glob
import os

from skimage import io, exposure

# define output directory
path_out = '../../output/suppl_fig2/b/'
for subdir in ['bf', 'flu', 'gt', 'pred']:
    os.makedirs(os.path.join(path_out, subdir), exist_ok=True)
    os.makedirs(os.path.join(path_out, 'zoomin', subdir), exist_ok=True)

# collect all images
bf_paths  = glob.glob('./images/*.png')
flu_paths  = glob.glob('./nuc/*.png')
gt_paths  = glob.glob('./gt/*.png')
pred_paths  = glob.glob('./pred/*.png')

# sort paths
bf_paths.sort()
flu_paths.sort()
gt_paths.sort()
pred_paths.sort()

# assign crop bounding boxes per image
crops = {
        '190905WW12_p0001_t00001_z001.png': [45,700, 547, 1202], #y_top, x_left, y_bottom, x_right, 502x502
        '190905WW12_p0011_t00001_z001.png': [42, 1132, 544, 1634],
        '190905WW12_p0013_t00001_z001.png': [1165, 1075, 1667, 1577],
        }

# load and crop images and collect min and max intensities
bp_wp_bf = {
        '190905WW12_p0001_t00001_z001.png': [5, 107],
        '190905WW12_p0011_t00001_z001.png': [5, 112],
        '190905WW12_p0013_t00001_z001.png': [5, 120],
        }
bp_wp_flu = {
        '190905WW12_p0001_t00001_z001.png': [0, 88],
        '190905WW12_p0011_t00001_z001.png': [0, 88],
        '190905WW12_p0013_t00001_z001.png': [0, 81],
        }
zoom_ins = {
        '190905WW12_p0001_t00001_z001.png': [140, 355, 240, 455], #x_left, y_top, x_right, y_bottom, 100x100
        '190905WW12_p0011_t00001_z001.png': [18, 208, 118, 308],
        '190905WW12_p0013_t00001_z001.png': [60, 355, 160, 455],
        }

for bf_path, flu_path, gt_path, pred_path in zip(bf_paths, flu_paths, gt_paths, pred_paths):
    bf = io.imread(bf_path)
    flu = io.imread(flu_path)
    gt = io.imread(gt_path)
    pred = io.imread(pred_path)

    # crop images
    bf = bf[crops[os.path.basename(bf_path)][0]:crops[os.path.basename(bf_path)][2],
            crops[os.path.basename(bf_path)][1]:crops[os.path.basename(bf_path)][3]]
    flu = flu[crops[os.path.basename(flu_path)][0]:crops[os.path.basename(flu_path)][2],
              crops[os.path.basename(flu_path)][1]:crops[os.path.basename(flu_path)][3]]
    gt = gt[crops[os.path.basename(gt_path)][0]:crops[os.path.basename(gt_path)][2],
            crops[os.path.basename(gt_path)][1]:crops[os.path.basename(gt_path)][3]]
    pred = pred[crops[os.path.basename(pred_path)][0]:crops[os.path.basename(pred_path)][2],
                crops[os.path.basename(pred_path)][1]:crops[os.path.basename(pred_path)][3]]

    # adjust min/max intensities
    img_name = os.path.basename(bf_path)
    bf = exposure.rescale_intensity(bf, in_range=(bp_wp_bf[img_name][0], bp_wp_bf[img_name][1]), out_range=(0, 255))
    flu = exposure.rescale_intensity(flu, in_range=(bp_wp_flu[img_name][0], bp_wp_flu[img_name][1]), out_range=(0, 255))
    
    # generate zoom in crops
    bf_zoom = bf[zoom_ins[img_name][1]:zoom_ins[img_name][3],zoom_ins[img_name][0]:zoom_ins[img_name][2]]
    flu_zoom = flu[zoom_ins[img_name][1]:zoom_ins[img_name][3],zoom_ins[img_name][0]:zoom_ins[img_name][2]]
    gt_zoom = gt[zoom_ins[img_name][1]:zoom_ins[img_name][3],zoom_ins[img_name][0]:zoom_ins[img_name][2]]
    pred_zoom = pred[zoom_ins[img_name][1]:zoom_ins[img_name][3],zoom_ins[img_name][0]:zoom_ins[img_name][2]]

    # save zoom in crops
    io.imsave(os.path.join(path_out, 'zoomin', 'bf', img_name), bf_zoom.astype('uint8'))
    io.imsave(os.path.join(path_out, 'zoomin', 'flu', img_name), flu_zoom.astype('uint8'))
    io.imsave(os.path.join(path_out, 'zoomin', 'gt', img_name), gt_zoom.astype('uint8'))
    io.imsave(os.path.join(path_out, 'zoomin', 'pred', img_name), pred_zoom.astype('uint8'))

    # draw white square around zoom_ins
    if img_name != '210930MA20_p0015_t00005_z001.png':
        x1 = zoom_ins[img_name][0] - 1
        x2 = zoom_ins[img_name][2] + 1
        y1 = zoom_ins[img_name][1] - 1
        y2 = zoom_ins[img_name][3] + 1

        x1 = max(x1, 0)
        x2 = min(x2, bf.shape[1] - 1)
        y1 = max(y1, 0)
        y2 = min(y2, bf.shape[0] - 1)

        bf[y1:y2,x1] = 255
        bf[y1:y2,x2] = 255
        bf[y1,x1:x2] = 255
        bf[y2,x1:x2] = 255

        flu[y1:y2,x1] = 255
        flu[y1:y2,x2] = 255
        flu[y1,x1:x2] = 255
        flu[y2,x1:x2] = 255

        gt[y1:y2,x1] = 255
        gt[y1:y2,x2] = 255
        gt[y1,x1:x2] = 255
        gt[y2,x1:x2] = 255

        pred[y1:y2,x1] = 255
        pred[y1:y2,x2] = 255
        pred[y1,x1:x2] = 255
        pred[y2,x1:x2] = 255

    # save crops
    io.imsave(os.path.join(path_out, 'bf', img_name), bf.astype('uint8'))
    io.imsave(os.path.join(path_out, 'flu', img_name), flu.astype('uint8'))
    io.imsave(os.path.join(path_out, 'gt', img_name), gt.astype('uint8'))
    io.imsave(os.path.join(path_out, 'pred', img_name), pred.astype('uint8'))

