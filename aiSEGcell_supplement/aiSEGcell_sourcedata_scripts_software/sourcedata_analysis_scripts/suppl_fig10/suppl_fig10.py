import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import io, exposure, measure
from skimage.color import label2rgb
from typing import Any, Dict, Tuple


def convert_to_greyscale(im):
    if im.shape[-1] == 3:
        im = im[:,:,0]
    return im

def load_data_from_row(row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any], float, float, float, float, Tuple]:
    raw_im = io.imread(row.iloc[0])
    labels_nuc = io.imread(row.iloc[1])
    labels_cell = io.imread(row.iloc[2])
    Xmin = row.iloc[3]
    Xmax = row.iloc[4]
    Ymin = row.iloc[5]
    Ymax = row.iloc[6]
    cell_idx = row.iloc[7]
    
    shape_val = raw_im.shape
    reset_img = np.zeros(shape_val)
    
    contours = measure.find_contours(labels_nuc)
    for i, contour in enumerate(contours):
        contour = np.array(contour, dtype=int)
        reset_img[contour[:,0],contour[:,1]] = 255
    labels_nuc = reset_img
    
    reset_img = np.zeros(shape_val)
    
    contours = measure.find_contours(labels_cell)
    for i, contour in enumerate(contours):
        contour = np.array(contour, dtype=int)
        reset_img[contour[:,0],contour[:,1]] = 255
    labels_cell = reset_img
    
    feature_name = row.iloc[8]
    feature_value = row.iloc[9]
    IOU = row.iloc[10]

    return raw_im, labels_nuc, labels_cell, Xmin, Xmax, Ymin, Ymax

# define output directory
path_out = '../output/suppl_fig10'
os.makedirs(path_out, exist_ok=True)

# get list of examples
DATASET_TABLE_IN = './napari_df_extreme_examples_crops.csv'

# load and crop images and collect min and max intensities
bp_wp_bf = {
        '181024TK20_p0002_t00012_z001.png': [0, 68],
        '181024TK20_p0002_t00049_z001.png': [0, 68],
        '181024TK20_p0009_t00035_z001.png': [0, 70],
        '181024TK20_p0009_t00107_z001.png': [0, 70],
        '181024TK20_p0017_t00061_z001.png': [0, 75],
        '181024TK20_p0017_t00062_z001.png': [0, 75],
        '181024TK20_p0017_t00112_z001.png': [0, 75],
        '200313SK20_p0010_t00025_z001.png': [68, 205],
        '200313SK20_p0020_t00006_z001.png': [44, 174],
        '200925SK30_p0013_t00099_z001.png': [99, 198],
        '200925SK30_p0013_t00122_z001.png': [109, 187],
        '200925SK30_p0013_t00184_z001.png': [109, 187],
        '200925SK30_p0042_t00089_z001.png': [104, 172],
        '200925SK30_p0042_t00118_z001.png': [104, 172],
        '200925SK30_p0042_t00119_z001.png': [104, 172],
        '210930MA20_p0001_t00066_z001.png': [65, 165],
        '210930MA20_p0003_t00013_z001.png': [47, 182],
        '210930MA20_p0003_t00024_z001.png': [47, 182],
        '210930MA20_p0003_t00033_z001.png': [47, 182],
        '210930MA20_p0007_t00026_z001.png': [60, 198],
        '210930MA20_p0013_t00004_z001.png': [49, 169],
        '210930MA20_p0013_t00047_z001.png': [49, 169],
        '210930MA20_p0013_t00074_z001.png': [49, 169],
        '210930MA20_p0015_t00046_z001.png': [36, 195],
        '210930MA20_p0019_t00070_z001.png': [60, 156],
        '220524DS30_p0005_t00072_z001.png': [52, 193],##
        '220524DS30_p0019_t00019_z001.png': [29, 195],
        '220524DS30_p0021_t00078_z001.png': [46, 161],
        '220524DS30_p0028_t00055_z001.png': [21, 156],
        '220524DS30_p0029_t00049_z001.png': [31, 200],
        '220524DS30_p0029_t00083_z001.png': [31, 200],
        '220524DS30_p0031_t00021_z001.png': [23, 133],
        '220524DS30_p0032_t00051_z001.png': [0, 156],
        }

df = pd.read_csv(DATASET_TABLE_IN)

for index in range(len(df)):
    rowi = df.iloc[index, :]
    img_name = os.path.basename(rowi.iloc[0])
    feature_name = rowi.iloc[8]
    feature_value = rowi.iloc[9]
    raw_im, labels_nuc, labels_cell, Xmin, Xmax, Ymin, Ymax = load_data_from_row(rowi)
    raw_im = convert_to_greyscale(raw_im)
    labels_nuc = convert_to_greyscale(labels_nuc)
    labels_cell = convert_to_greyscale(labels_cell)

    # adjust intensities (where necessary)
    if not("Intensity" in feature_name or "Gradient" in feature_name or "focus" in feature_name or "cut_offness" in feature_name):
        print(feature_name, "  intensity changed")
        raw_im = exposure.rescale_intensity(raw_im, in_range=(bp_wp_bf[img_name][0], bp_wp_bf[img_name][1]), out_range=(0, 255)) 
        raw_im = raw_im.astype('uint8')

    # set square size (30 px extra)
    lenX = (Xmax-Xmin)
    lenY = (Ymax-Ymin)
    lenAll = max([lenX,lenY]) + 10

    addX = int(math.floor((lenAll - lenX)/2))
    Xmax = int(Xmax + addX)
    Xmin = int(Xmin - (lenAll - lenX - addX))

    
    addY= int(math.floor((lenAll - lenY)/2))
    Ymax = int(Ymax + addY)
    Ymin = int(Ymin - (lenAll - lenY - addY))
    
    if Xmin < 0:
        Xmax = Xmax + abs(Xmin)
        Xmin = 0
    if Ymin < 0:
        Ymax = Ymax + abs(Ymin)
        Ymin = 0

    if Xmax > raw_im.shape[0]:
        Xmin = Xmin - (Xmax-raw_im.shape[0])
        Xmax = raw_im.shape[0]    
    if Ymax > raw_im.shape[0]:
        Ymin = Ymin - (Ymax-raw_im.shape[0])
        Ymax = raw_im.shape[0]  

    # overlay 
    if ("cell_mask" in feature_name) or ("focus" in feature_name):
        labels_cell = measure.label(labels_cell)
        image_label_overlay = label2rgb(labels_cell, image=raw_im, bg_label=0, colors=[(222/255, 209/255, 42/255)])
        
    # elif "multinucleated" in feature_name:
    #     labels_cell = measure.label(labels_cell)
    #     image_label_overlay = label2rgb(labels_cell, image=raw_im, bg_label=0, colors=[(222/255, 209/255, 42/255)])
    #     labels_nuc = measure.label(labels_nuc)
    #     image_label_overlay = label2rgb(labels_nuc, image=image_label_overlay, bg_label=0, colors=[(246/255, 122/255, 122/255)]) 
        
    else:
        labels_nuc = measure.label(labels_nuc)
        image_label_overlay = label2rgb(labels_nuc, image=raw_im, bg_label=0, colors=[(246/255, 122/255, 122/255)])          

    # crop 
    image_label_overlay = image_label_overlay[Ymin:Ymax,Xmin:Xmax]
    image_label_overlay = (255*image_label_overlay).astype('uint8') 

    # save crops
    io.imsave(os.path.join(path_out, feature_name+"_"+str(round(feature_value,2))+"_"+img_name), image_label_overlay)
