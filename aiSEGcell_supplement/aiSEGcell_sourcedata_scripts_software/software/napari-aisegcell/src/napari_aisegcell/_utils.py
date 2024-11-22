#######################################################################################################################
# Utility functions for the plugin.                                                                                   #
# © 2020-2024, ETH Zurich                                                                                             #
# Author:               Daniel Schirmacher, Timm Schroeder                                                            #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                               #
# Python Version:       3.8.7                                                                                         #
# PyTorch Version:      1.7.1                                                                                         #
# PyTorch Lightning Version: 1.5.9                                                                                    #
#######################################################################################################################

import functools
import os
from typing import List

import numpy as np
import torch
from psygnal import Signal
from skimage.measure import label
from skimage.morphology import (
    dilation,
    erosion,
    remove_small_holes,
    remove_small_objects,
    square,
)
from torchvision import transforms


# taken from stardist-napari (https://github.com/stardist/stardist-napari)
def change_handler(*widgets, init=True, debug=False):
    def decorator_change_handler(handler):
        @functools.wraps(handler)
        def wrapper(*args):
            source = Signal.sender()
            emitter = Signal.current_emitter()
            if debug:
                # print(f"{emitter}: {source} = {args!r}")
                print(f"{str(emitter.name).upper()}: {source.name} = {args!r}")
            return handler(*args)

        for widget in widgets:
            widget.changed.connect(wrapper)
            if init:
                widget.changed(widget.value)
        return wrapper

    return decorator_change_handler


def check_order(l1: List[str], l2: List[str]) -> bool:
    """
    Check if 2 lists of paths have identical names and order of files.
    """
    assert len(l1) == len(l2), "l1 and l2 must be of same length."
    assert all(isinstance(i, str) for i in l1), "l1 must be a List[str]"
    assert all(isinstance(i, str) for i in l2), "l2 must be a List[str]"

    l1 = [f.split(os.path.sep)[-1] for f in l1]
    l2 = [f.split(os.path.sep)[-1] for f in l2]

    return l1 == l2


def rename_duplicates(s: List[str]) -> List[str]:
    """
    Add IDs to duplicate file names. From
    https://stackoverflow.com/a/30651843/2437514
    """
    assert isinstance(s, list), f's must be of type list, but is "{type(s)}".'
    assert all(isinstance(i, str) for i in s), "s must be a List[str]"

    dups = {}

    for i, val in enumerate(s):
        if val not in dups:
            # Store index of first occurrence and occurrence value
            dups[val] = [i, 1]
        else:
            # Special case for first occurrence
            if dups[val][1] == 1:
                s_0_split = s[dups[val][0]].split(".")
                s_0_split[-2] += f"_{dups[val][1]}"
                s_0 = ".".join(s_0_split[:-1] + s_0_split[-1:])
                s[dups[val][0]] = s_0

            # Increment occurrence value, index value doesn't matter anymore
            dups[val][1] += 1

            # Use stored occurrence value
            s_i_split = s[i].split(".")
            s_i_split[-2] += f"_{dups[val][1]}"
            s_i = ".".join(s_i_split[:-1] + s_i_split[-1:])
            s[i] = s_i

    return s


def _preprocess(img: np.ndarray, device: str) -> torch.FloatTensor:
    assert (
        len(img.shape) == 2
    ), f"Input image is expected to be 2D, but is {len(img.shape)}D."

    # convert input image to tensor
    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=2)
    img = img.transpose((3, 2, 0, 1))  # (batch, colorchannel, height, width)

    # check for np.uint16 (not supported by torch) format
    if img.dtype == np.uint16:
        max_intensity = 65535
        img = img.astype(np.int32)
    elif img.dtype == np.uint8:
        max_intensity = 255
    else:
        max_intensity = 255
        raise Warning(
            "Image is expected to be in {np.uint8, np.uint16}, but "
            f'is "{img.dtype}". Integer overflow may cause '
            "erroneous segmentations."
        )

    img_t = torch.from_numpy(img).type(torch.FloatTensor)

    # normalize tensor
    transform = transforms.Normalize(0, max_intensity)
    img_t = transform(img_t)
    img_t = img_t.to(device)

    return img_t


def _postprocess(
    mask: np.ndarray,
    remove_holes: int,
    size_min: int,
    size_max: int,
    instance_segmentation: bool,
    dilate: int,
) -> np.ndarray:
    assert (
        len(mask.shape) == 2
    ), f"Mask is expected to be 2D, but is {len(mask.shape)}D."
    mask_cp = mask.copy()

    # remove holes and objects that violate size restrictions
    arr = mask_cp > 0

    if remove_holes > 1:
        arr = remove_small_holes(arr, remove_holes)

    if size_min > 1:
        arr = remove_small_objects(arr, size_min)

    mask_cp[arr] = 255
    mask_cp[~arr] = 0

    mask_cp = label(mask_cp)

    if size_max < 1000000:
        ids, counts = np.unique(mask_cp, return_counts=True)
        ids = ids[1:]
        counts = counts[1:]
        ids = ids[counts > size_max]
        mask_cp[np.isin(mask_cp, ids)] = 0

    if not instance_segmentation:
        mask_cp[mask_cp > 0] = 255

    # dilate/erode mask
    if dilate > 0:
        mask_cp = dilation(mask_cp, square(1 + 2 * dilate))
    elif dilate < 0:
        mask_cp = erosion(mask_cp, square(1 - 2 * dilate))

    return mask_cp
