import glob
import os
import numpy as np 

from skimage import io, exposure

# define output directory
path_out = '../output/suppl_fig13/'
for subdir in ['bf', 'flu', 'gt', 'pred', 'gradcam']:
    os.makedirs(os.path.join(path_out, subdir), exist_ok=True)
    os.makedirs(os.path.join(path_out, 'zoomin', subdir), exist_ok=True)

# collect all images
bf_paths  = glob.glob('./images/*/*.png')
gradcam_paths  = glob.glob('./gradcam/*/*.png')
flu_paths  = glob.glob('./nuc/*/*.png')
gt_paths  = glob.glob('./gt/*/*.png')
pred_paths  = glob.glob('./pred/*/*.png')

# sort paths
bf_paths.sort()
gradcam_paths.sort()
flu_paths.sort()
gt_paths.sort()
pred_paths.sort()

# assign crop bounding boxes per image
crops = {
        '190905WW12_p0001_t00001_z001.png': [45,700, 547, 1202], #y_top, x_left, y_bottom, x_right, 502x502
        '190905WW12_p0011_t00001_z001.png': [42, 1132, 544, 1634],
        '190905WW12_p0013_t00001_z001.png': [1165, 1075, 1667, 1577],
        '181024TK20_p0002_t00006_z001.png': [390, 315, 646, 571],
        '200313SK20_p0001_t00074_z001.png': [520, 169, 776, 425],
        '200925SK30_p0030_t00001_z001.png': [25, 420, 537, 932],
        '210930MA20_p0002_t00023_z001.png': [510, 450, 766, 706],
        '220524DS30_p0004_t00020_z001.png': [840, 745, 1352, 1257],
        '210824YZ12_p0004_t00001_z001.png': [435, 156, 1287, 1008], #y_top, x_left, y_bottom, x_right, #852x852
        '210824YZ12_p0014_t00001_z001.png': [624, 401, 1476, 1253],
        '210824YZ12_p0027_t00001_z001.png': [468, 657, 1320, 1509], 
        '170601SH11_p0023_t00001_z001.png': [746, 1229, 1248, 1731], #y_top, x_left, y_bottom, x_right, 502x502
        '170601SH11_p0026_t00001_z001.png': [419, 1080, 921, 1582],
        '170601SH11_p0027_t00001_z001.png': [285, 502, 787, 1004],
        }

# load and crop images and collect min and max intensities
bp_wp_bf = {
        '190905WW12_p0001_t00001_z001.png': [5, 107],
        '190905WW12_p0011_t00001_z001.png': [5, 112],
        '190905WW12_p0013_t00001_z001.png': [5, 120],
        '181024TK20_p0002_t00006_z001.png': [10, 50],
        '200313SK20_p0001_t00074_z001.png': [40, 100],
        '200925SK30_p0030_t00001_z001.png': [120, 180],
        '210930MA20_p0002_t00023_z001.png': [60, 140],
        '220524DS30_p0004_t00020_z001.png': [70, 130],
        '210824YZ12_p0004_t00001_z001.png': [55, 138],
        '210824YZ12_p0014_t00001_z001.png': [75, 133],
        '210824YZ12_p0027_t00001_z001.png': [49, 135],
        '170601SH11_p0023_t00001_z001.png': [70, 117],
        '170601SH11_p0026_t00001_z001.png': [65, 122],
        '170601SH11_p0027_t00001_z001.png': [65, 120],
        }
bp_wp_flu = {
        '190905WW12_p0001_t00001_z001.png': [0, 88],
        '190905WW12_p0011_t00001_z001.png': [0, 88],
        '190905WW12_p0013_t00001_z001.png': [0, 81],
        '181024TK20_p0002_t00006_z001.png': [0, 100],
        '200313SK20_p0001_t00074_z001.png': [0, 130],
        '200925SK30_p0030_t00001_z001.png': [30, 110],
        '210930MA20_p0002_t00023_z001.png': [0, 100],
        '220524DS30_p0004_t00020_z001.png': [0, 140],
        '210824YZ12_p0004_t00001_z001.png': [10, 23],
        '210824YZ12_p0014_t00001_z001.png': [0, 143],
        '210824YZ12_p0027_t00001_z001.png': [10, 81],
        '170601SH11_p0023_t00001_z001.png': [3, 31],
        '170601SH11_p0026_t00001_z001.png': [3, 36],
        '170601SH11_p0027_t00001_z001.png': [3, 29],
        }
zoom_ins = {
        '190905WW12_p0001_t00001_z001.png': [140, 355, 240, 455], #x_left, y_top, x_right, y_bottom, 100x100
        '190905WW12_p0011_t00001_z001.png': [18, 208, 118, 308],
        '190905WW12_p0013_t00001_z001.png': [60, 355, 160, 455],
        '181024TK20_p0002_t00006_z001.png': [0, 107, 32, 139], # 32x32
        '200313SK20_p0001_t00074_z001.png': [180, 155, 228, 203], # 48x48
        '200925SK30_p0030_t00001_z001.png': [30, 30, 126, 126], # 96x96
        '210930MA20_p0002_t00023_z001.png': [212, 176, 252, 216], # 40x40
        '220524DS30_p0004_t00020_z001.png': [410, 260, 490, 340], # 80x80
        '210824YZ12_p0004_t00001_z001.png': [370, 510, 670, 810], #x_left, y_top, x_right, y_bottom, 300x300
        '210824YZ12_p0014_t00001_z001.png': [99, 472, 399, 772],
        '210824YZ12_p0027_t00001_z001.png': [249, 528, 549, 828], 
        '170601SH11_p0023_t00001_z001.png': [71, 326, 171, 426], #x_left, y_top, x_right, y_bottom, 100x100
        '170601SH11_p0026_t00001_z001.png': [98, 326, 198, 426],
        '170601SH11_p0027_t00001_z001.png': [91, 320, 191, 420],
        }

for bf_path, flu_path, gt_path, pred_path, gradcam_path in zip(bf_paths, flu_paths, gt_paths, pred_paths, gradcam_paths):
    bf = io.imread(bf_path)
    flu = io.imread(flu_path)
    gt = io.imread(gt_path)
    pred = io.imread(pred_path)
    gradcam = io.imread(gradcam_path)

    # crop images
    bf = bf[crops[os.path.basename(bf_path)][0]:crops[os.path.basename(bf_path)][2],
            crops[os.path.basename(bf_path)][1]:crops[os.path.basename(bf_path)][3]]
    flu = flu[crops[os.path.basename(flu_path)][0]:crops[os.path.basename(flu_path)][2],
              crops[os.path.basename(flu_path)][1]:crops[os.path.basename(flu_path)][3]]
    gt = gt[crops[os.path.basename(gt_path)][0]:crops[os.path.basename(gt_path)][2],
            crops[os.path.basename(gt_path)][1]:crops[os.path.basename(gt_path)][3]]
    pred = pred[crops[os.path.basename(pred_path)][0]:crops[os.path.basename(pred_path)][2],
                crops[os.path.basename(pred_path)][1]:crops[os.path.basename(pred_path)][3]]
    gradcam = gradcam[crops[os.path.basename(gradcam_path)][0]:crops[os.path.basename(gradcam_path)][2],
                crops[os.path.basename(gradcam_path)][1]:crops[os.path.basename(gradcam_path)][3]]

    # adjust min/max intensities
    img_name = os.path.basename(bf_path)
    bf = exposure.rescale_intensity(bf, in_range=(bp_wp_bf[img_name][0], bp_wp_bf[img_name][1]), out_range=(0, 255))
    flu = exposure.rescale_intensity(flu, in_range=(bp_wp_flu[img_name][0], bp_wp_flu[img_name][1]), out_range=(0, 255))
    
    # generate zoom in crops
    bf_zoom = bf[zoom_ins[img_name][1]:zoom_ins[img_name][3],zoom_ins[img_name][0]:zoom_ins[img_name][2]]
    flu_zoom = flu[zoom_ins[img_name][1]:zoom_ins[img_name][3],zoom_ins[img_name][0]:zoom_ins[img_name][2]]
    gt_zoom = gt[zoom_ins[img_name][1]:zoom_ins[img_name][3],zoom_ins[img_name][0]:zoom_ins[img_name][2]]
    pred_zoom = pred[zoom_ins[img_name][1]:zoom_ins[img_name][3],zoom_ins[img_name][0]:zoom_ins[img_name][2]]
    gradcam_zoom = gradcam[zoom_ins[img_name][1]:zoom_ins[img_name][3],zoom_ins[img_name][0]:zoom_ins[img_name][2]]

    # save zoom in crops
    io.imsave(os.path.join(path_out, 'zoomin', 'bf', img_name), bf_zoom.astype('uint8'))
    io.imsave(os.path.join(path_out, 'zoomin', 'flu', img_name), flu_zoom.astype('uint8'))
    io.imsave(os.path.join(path_out, 'zoomin', 'gt', img_name), gt_zoom.astype('uint8'))
    io.imsave(os.path.join(path_out, 'zoomin', 'pred', img_name), pred_zoom.astype('uint8'))
    io.imsave(os.path.join(path_out, 'zoomin', 'gradcam', img_name), gradcam_zoom)

    # draw white square around zoom_ins
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

    gradcam[y1:y2,x1] = 255
    gradcam[y1:y2,x2] = 255
    gradcam[y1,x1:x2] = 255
    gradcam[y2,x1:x2] = 255

    # save crops
    io.imsave(os.path.join(path_out, 'bf', img_name), bf.astype('uint8'))
    io.imsave(os.path.join(path_out, 'flu', img_name), flu.astype('uint8'))
    io.imsave(os.path.join(path_out, 'gt', img_name), gt.astype('uint8'))
    io.imsave(os.path.join(path_out, 'pred', img_name), pred.astype('uint8'))
    io.imsave(os.path.join(path_out, 'gradcam', img_name), gradcam)
