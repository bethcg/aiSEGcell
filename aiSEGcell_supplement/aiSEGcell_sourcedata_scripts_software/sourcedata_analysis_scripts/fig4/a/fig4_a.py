import glob
import os

from skimage import io, exposure


# define output directory
path_out = '../../output/fig4/a'
os.makedirs(path_out, exist_ok=True)

# collect all images
path_data  = glob.glob('./data/*.png')

# assign crop bounding boxes per image
crops = {
        '230624DS30_p0018_t00007_z001_w00.png': [890, 1485, 950, 1545],
        '230624DS30_p0018_t00007_z001_w01.png': [890, 1485, 950, 1545],
        '230624DS30_p0018_t00007_z001_w02.png': [890, 1485, 950, 1545],
        '230624DS30_p0018_t00007_z001_w03.png': [890, 1485, 950, 1545],
        '230624DS30_p0018_t00007_z001_w04.png': [890, 1485, 950, 1545],
        '230624DS30_p0018_t00030_z001_w00.png': [890, 1465, 950, 1525],
        '230624DS30_p0018_t00030_z001_w01.png': [890, 1465, 950, 1525],
        '230624DS30_p0018_t00030_z001_w02.png': [890, 1465, 950, 1525],
        '230624DS30_p0018_t00030_z001_w03.png': [890, 1465, 950, 1525],
        '230624DS30_p0018_t00030_z001_w04.png': [890, 1465, 950, 1525],
        '230624DS30_p0018_t00157_z001_w00.png': [870, 1270, 950, 1350],
        '230624DS30_p0018_t00157_z001_w01.png': [870, 1270, 950, 1350],
        '230624DS30_p0018_t00157_z001_w02.png': [870, 1270, 950, 1350],
        '230624DS30_p0018_t00157_z001_w03.png': [870, 1270, 950, 1350],
        '230624DS30_p0018_t00157_z001_w04.png': [870, 1270, 950, 1350],
        }

# load and crop images and collect min and max intensities
bp_wp = {
        '230624DS30_p0018_t00007_z001_w00.png': [80, 160],
        '230624DS30_p0018_t00007_z001_w01.png': [15, 80],
        '230624DS30_p0018_t00007_z001_w02.png': [5, 40],
        '230624DS30_p0018_t00007_z001_w03.png': [0, 255],
        '230624DS30_p0018_t00007_z001_w04.png': [5, 50],
        '230624DS30_p0018_t00030_z001_w00.png': [80, 160],
        '230624DS30_p0018_t00030_z001_w01.png': [15, 80],
        '230624DS30_p0018_t00030_z001_w02.png': [5, 40],
        '230624DS30_p0018_t00030_z001_w03.png': [0, 255],
        '230624DS30_p0018_t00030_z001_w04.png': [5, 50],
        '230624DS30_p0018_t00157_z001_w00.png': [80, 160],
        '230624DS30_p0018_t00157_z001_w01.png': [15, 80],
        '230624DS30_p0018_t00157_z001_w02.png': [5, 40],
        '230624DS30_p0018_t00157_z001_w03.png': [0, 255],
        '230624DS30_p0018_t00157_z001_w04.png': [5, 50],
        }

for path_img in path_data:
    img = io.imread(path_img)

    # crop images
    img_name = os.path.basename(path_img)
    img = img[crops[img_name][0]:crops[img_name][2],
              crops[img_name][1]:crops[img_name][3]]

    # adjust min/max intensities
    img = exposure.rescale_intensity(img, in_range=(bp_wp[img_name][0], bp_wp[img_name][1]), out_range=(0, 255))
    
    # save crops
    io.imsave(os.path.join(path_out, img_name), img.astype('uint8'))
