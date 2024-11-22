########################################################################################################################
# This script runs napari and implements a button to load image pairs from a df (adapted from Kevin Yamauchi).         #
# Author: Daniel Schirmacher                                                                                           #
# Date: 22.12.2021                                                                                                     #
# Python: 3.8.6                                                                                                        #
########################################################################################################################
import glob
import os
from typing import Any, Dict, Tuple
import napari
import numpy as np
import pandas as pd
from skimage import io, measure
from vispy import scene, app
import warnings

# accessing the zoom in the camera gives warnings, ignore them
warnings.filterwarnings("ignore")

DATASET_TABLE_IN = './napari_df_extreme_examples_crops.csv'

scales = {'181024TK20': (0.62,0.62), '200313SK20': (0.62,0.62), '200925SK30': (0.34,0.34), '210930MA20': (0.62,0.62), '220524DS30': (0.34,0.34)}


def load_data_from_row(row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any], float, float, float, float, Tuple]:
    raw_im = io.imread(row.iloc[0])
    labels_nuc = io.imread(row.iloc[1])
    labels_cell = io.imread(row.iloc[2])
    Xmin = row.iloc[3]
    Xmax = row.iloc[4]
    Ymin = row.iloc[5]
    Ymax = row.iloc[6]
    
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
    
    print(feature_name, ': ', feature_value)
    print('IOU: ', IOU)

    print(row.iloc[1].split('/')[-1])
    print('cell idx:', row.iloc[7])
    
    metadata_nuc = {
                'fpath': row.iloc[1]
            }
    metadata_cell = {
                'fpath': row.iloc[2]
            }
    movie_name = row.iloc[1].split('/')[-1].split('_')[0]
    scale = scales[movie_name]

    return raw_im, labels_nuc, labels_cell, metadata_nuc, metadata_cell, Xmin, Xmax, Ymin, Ymax, scale

def zoom(Xmin: float, Xmax: float, Ymin: float, Ymax: float, scale: Tuple) -> Dict:
    
    # scale coordinates to pixel size
    Xmin = int(scale[0]*Xmin)
    Xmax = int(scale[0]*Xmax)
    Ymin = int(scale[1]*Ymin)
    Ymax = int(scale[1]*Ymax)
    
    rect = ((Xmin, Ymin), (Xmax-Xmin, Ymax-Ymin))
    state = {'rect': rect}
    
    return state


# initialize loading
index = 0
df = pd.read_csv(DATASET_TABLE_IN)

# get the initial data
initial_row = df.iloc[index, :]
raw_im, labels_nuc, labels_cell, metadata_nuc, metadata_cell, Xmin, Xmax, Ymin, Ymax, scale = load_data_from_row(initial_row)

# make the viewer
viewer = napari.Viewer()
viewer.window.qt_viewer.view.rect._pos = (500,500)

viewer.add_image(raw_im, name='raw_im', scale=scale)
viewer.add_labels(labels_nuc.astype(int), opacity=0.30, name='labels_nuc', metadata=metadata_nuc, color={255:'#F67A7A'}, scale=scale)
viewer.add_labels(labels_cell.astype(int), opacity=0.20, name='labels_cell', metadata=metadata_cell, color={255:'#DED12A'}, scale=scale)

viewer.scale_bar.visible = True
viewer.scale_bar.unit = "um"

# zoom into cell
state = zoom(Xmin, Xmax, Ymin, Ymax, scale)
viewer.window.qt_viewer.view.camera.set_state(state)

@viewer.bind_key('f')
def flag_file(event_viewer=None, event=None):
    
    if df.loc[index,'flag'] == 0:
        df.loc[index,'flag'] = 1  
        print('flagged as:', 1)
        
    else:
        df.loc[index,'flag'] = 0  
        print('flagged as:', 0)
        
    df.to_csv(DATASET_TABLE_IN, index=False)   
    
    
@viewer.bind_key('s')
def take_screenshot(event_viewer=None, event=None):
    row = df.iloc[index, :]
    feature_name = row.iloc[8]
    feature_value = row.iloc[9]
    IOU = row.iloc[10]
    file_name = str(feature_name) + '_' + str(round(feature_value,2)) + '_IOU_' + str(round(IOU,2)) + '.png'
    viewer.screenshot(file_name)           


@viewer.bind_key('n')
def load_next_data(event_viewer=None, event=None):
    global index
    global df
    

    # load the new data
    index += 1
    print(index)
    if index%200 == 0:
        print(index)

    # continue with next row
    try:
        next_row = df.iloc[index]
    except IndexError:
        raise IndexError('The list of image triplets is empty')
        return

    raw_im, labels_nuc, labels_cell, metadata_nuc, metadata_cell, Xmin, Xmax, Ymin, Ymax, scale = load_data_from_row(next_row)

    viewer.layers['raw_im'].data = raw_im
    viewer.layers['raw_im'].scale = scale
    viewer.layers['labels_nuc'].data = labels_nuc.astype(int)
    viewer.layers['labels_nuc'].metadata = metadata_nuc
    viewer.layers['labels_nuc'].scale = scale
    viewer.layers['labels_cell'].data = labels_cell.astype(int)
    viewer.layers['labels_cell'].metadata = metadata_cell
    viewer.layers['labels_cell'].scale = scale
    
    # zoom into cell
    state = zoom(Xmin, Xmax, Ymin, Ymax, scale)
    viewer.window.qt_viewer.view.camera.set_state(state)

    
@viewer.bind_key('b')
def load_next_data(event_viewer=None, event=None):
    global index
    global df

    # load the new data
    index -= 1
    print(index)
    
    # continue with previous row
    try:
        next_row = df.loc[index]
    except IndexError:
        raise IndexError('The list of image triplets is empty')
        return

    raw_im, labels_nuc, labels_cell, metadata_nuc, metadata_cell, Xmin, Xmax, Ymin, Ymax, scale = load_data_from_row(next_row)

    viewer.layers['raw_im'].data = raw_im
    viewer.layers['raw_im'].scale = scale
    viewer.layers['labels_nuc'].data = labels_nuc.astype(int)
    viewer.layers['labels_nuc'].metadata = metadata_nuc
    viewer.layers['labels_nuc'].scale = scale
    viewer.layers['labels_cell'].data = labels_cell.astype(int)
    viewer.layers['labels_cell'].metadata = metadata_cell
    viewer.layers['labels_cell'].scale = scale   
    
    # zoom into cell
    state = zoom(Xmin, Xmax, Ymin, Ymax, scale)
    viewer.window.qt_viewer.view.camera.set_state(state)
    

napari.run()
