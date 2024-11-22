import glob
import os

from skimage import io, exposure

# define output directory
path_out = '../output/suppl_fig9'
for subdir in ['bf', 'gt']:
    os.makedirs(os.path.join(path_out, subdir), exist_ok=True)
    os.makedirs(os.path.join(path_out, 'zoomin', subdir), exist_ok=True)

# collect all images
bf_paths  = glob.glob('./images/*.png')
gt_paths  = glob.glob('./gt/*.png')

# sort paths
bf_paths.sort()
gt_paths.sort()

# assign crop bounding boxes per image
crops = {
        '200313SK20_p0011_t00037_z001.png': [0,0, 1024, 1042], #y_top, x_left, y_bottom, x_right, 502x502
        }

# load and crop images and collect min and max intensities
bp_wp_bf = {
        '200313SK20_p0011_t00037_z001.png': [81, 156],
        }

zoom_ins = {
        '200313SK20_p0011_t00037_z001.png': [601, 176, 821, 396], #x_left, y_top, x_right, y_bottom, 100x100
        }

for bf_path, gt_path in zip(bf_paths, gt_paths):
    bf = io.imread(bf_path)
    gt = io.imread(gt_path)

    # crop images
    bf = bf[crops[os.path.basename(bf_path)][0]:crops[os.path.basename(bf_path)][2],
            crops[os.path.basename(bf_path)][1]:crops[os.path.basename(bf_path)][3]]
    gt = gt[crops[os.path.basename(gt_path)][0]:crops[os.path.basename(gt_path)][2],
            crops[os.path.basename(gt_path)][1]:crops[os.path.basename(gt_path)][3]]

    # adjust min/max intensities
    img_name = os.path.basename(bf_path)
    bf = exposure.rescale_intensity(bf, in_range=(bp_wp_bf[img_name][0], bp_wp_bf[img_name][1]), out_range=(0, 255))
    
    # generate zoom in crops
    bf_zoom = bf[zoom_ins[img_name][1]:zoom_ins[img_name][3],zoom_ins[img_name][0]:zoom_ins[img_name][2]]
    gt_zoom = gt[zoom_ins[img_name][1]:zoom_ins[img_name][3],zoom_ins[img_name][0]:zoom_ins[img_name][2]]

    # save zoom in crops
    io.imsave(os.path.join(path_out, 'zoomin', 'bf', img_name), bf_zoom.astype('uint8'))
    io.imsave(os.path.join(path_out, 'zoomin', 'gt', img_name), gt_zoom.astype('uint8'))

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

    gt[y1:y2,x1] = 255
    gt[y1:y2,x2] = 255
    gt[y1,x1:x2] = 255
    gt[y2,x1:x2] = 255

    # save crops
    io.imsave(os.path.join(path_out, 'bf', img_name), bf.astype('uint8'))
    io.imsave(os.path.join(path_out, 'gt', img_name), gt.astype('uint8'))
