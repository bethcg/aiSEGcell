import ipdb
import glob
import os

from skimage import io, exposure

# define output directory
path_out = '../output/suppl_fig3/'
for subdir in ['bf', 'flu', 'gt', 'pred']:
    os.makedirs(os.path.join(path_out, subdir), exist_ok=True)
    os.makedirs(os.path.join(path_out, 'zoomin', subdir), exist_ok=True)

# collect all images
bf_paths  = glob.glob('./brightfield/*.png')
flu_paths  = glob.glob('./nucleus/*.png')
gt_paths  = glob.glob('./gt/*.png')
pred_paths  = glob.glob('./pred/*.png')

# sort paths
bf_paths.sort()
flu_paths.sort()
gt_paths.sort()
pred_paths.sort()

# assign crop bounding boxes per image
crops = {
        '240417YZ15_p0026_t00001_z001.png': [500, 8, 774, 282], # 274x274
        '240417YZ15_p0050_t00001_z001.png': [0, 0, 274, 274],
        '240418YZ16_p0016_t00001_z001.png': [747, 175, 1021, 449],
        }

# load and crop images and collect min and max intensities
bp_wp_bf = {
        '240417YZ15_p0026_t00001_z001.png': [50, 175],
        '240417YZ15_p0050_t00001_z001.png': [50, 175],
        '240418YZ16_p0016_t00001_z001.png': [50, 175],
        }
bp_wp_flu = {
        '240417YZ15_p0026_t00001_z001.png': [20, 180],
        '240417YZ15_p0050_t00001_z001.png': [20, 180],
        '240418YZ16_p0016_t00001_z001.png': [20, 180],
        }
zoom_ins = {
        '240417YZ15_p0026_t00001_z001.png': [80, 191, 160, 271], # 80x80
        '240417YZ15_p0050_t00001_z001.png': [14, 100, 94, 180], # 80x80
        '240418YZ16_p0016_t00001_z001.png': [93, 115, 173, 195], # 80x80
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
    bf_zoom = bf[zoom_ins[img_name][0]:zoom_ins[img_name][2],zoom_ins[img_name][1]:zoom_ins[img_name][3]]
    flu_zoom = flu[zoom_ins[img_name][0]:zoom_ins[img_name][2],zoom_ins[img_name][1]:zoom_ins[img_name][3]]
    gt_zoom = gt[zoom_ins[img_name][0]:zoom_ins[img_name][2],zoom_ins[img_name][1]:zoom_ins[img_name][3]]
    pred_zoom = pred[zoom_ins[img_name][0]:zoom_ins[img_name][2],zoom_ins[img_name][1]:zoom_ins[img_name][3]]

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

        bf[x1:x2,y1] = 255
        bf[x1:x2,y2] = 255
        bf[x1,y1:y2] = 255
        bf[x2,y1:y2] = 255

        flu[x1:x2,y1] = 255
        flu[x1:x2,y2] = 255
        flu[x1,y1:y2] = 255
        flu[x2,y1:y2] = 255

        gt[x1:x2,y1] = 255
        gt[x1:x2,y2] = 255
        gt[x1,y1:y2] = 255
        gt[x2,y1:y2] = 255

        pred[x1:x2,y1] = 255
        pred[x1:x2,y2] = 255
        pred[x1,y1:y2] = 255
        pred[x2,y1:y2] = 255

    # save crops
    io.imsave(os.path.join(path_out, 'bf', img_name), bf.astype('uint8'))
    io.imsave(os.path.join(path_out, 'flu', img_name), flu.astype('uint8'))
    io.imsave(os.path.join(path_out, 'gt', img_name), gt.astype('uint8'))
    io.imsave(os.path.join(path_out, 'pred', img_name), pred.astype('uint8'))

