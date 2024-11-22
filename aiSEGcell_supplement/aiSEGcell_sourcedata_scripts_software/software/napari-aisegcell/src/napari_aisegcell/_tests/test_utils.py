#######################################################################################################################
# Tests of utility functions.                                                                                         #
# © 2020-2024, ETH Zurich                                                                                             #
# Author:               Daniel Schirmacher, Timm Schroeder                                                            #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                               #
# Python Version:       3.8.7                                                                                         #
# PyTorch Version:      1.7.1                                                                                         #
# PyTorch Lightning Version: 1.5.9                                                                                    #
#######################################################################################################################

import os

import numpy as np
import pytest
import torch

from napari_aisegcell._utils import (
    _postprocess,
    _preprocess,
    check_order,
    rename_duplicates,
)


def test_check_order():
    l1 = ["a", "b", "c"]
    l2 = ["b", "a", "c"]
    l3 = ["a", "b"]

    # add directories
    l1 = [os.path.join("some_path", f) for f in l1]
    l2 = [os.path.join("some_path", f) for f in l2]
    l3 = [os.path.join("some_path", f) for f in l3]

    assert check_order(l1, l1)

    with pytest.raises(AssertionError):
        assert check_order(l1, l2)

    with pytest.raises(AssertionError):
        check_order(l1, l3)


def test_rename_duplicates():
    l1 = ["a.png", "b.png", "c.png"]
    l2 = ["a.png", "b.png", "a.png"]
    l1 = [os.path.join("some_path", f) for f in l1]
    l2 = [os.path.join("some_path", f) for f in l2]

    l1_expected = l1
    l2_expected = ["a_1.png", "b.png", "a_2.png"]
    l2_expected = [os.path.join("some_path", f) for f in l2_expected]

    assert rename_duplicates(l1) == l1_expected
    assert rename_duplicates(l2) == l2_expected


def test_preprocess():
    img = np.array([[0, 255], [0, 0]]).astype(np.uint8)
    img_16 = img.astype(np.uint16)
    img_16[0, 1] = 65535

    img_t = _preprocess(img, device="cpu")
    img_16_t = _preprocess(img, device="cpu")

    img_t_expected = torch.FloatTensor([[[[0.0, 1.0], [0.0, 0.0]]]])

    assert torch.all(img_t == img_t_expected)
    assert torch.all(img_16_t == img_t_expected)

    with pytest.raises(AssertionError):
        _preprocess(np.zeros((1, 2, 2)), device="cpu")


def test_postprocess():
    mask = np.array(
        [[0, 255, 0, 255], [0, 0, 0, 0], [255, 255, 255, 0], [255, 0, 255, 0]]
    )

    mask1 = _postprocess(
        mask,
        remove_holes=2,
        size_min=1,
        size_max=100000,
        instance_segmentation=True,
        dilate=0,
    )

    mask2 = _postprocess(
        mask,
        remove_holes=0,
        size_min=2,
        size_max=100000,
        instance_segmentation=False,
        dilate=1,
    )

    mask3 = _postprocess(
        mask,
        remove_holes=0,
        size_min=1,
        size_max=2,
        instance_segmentation=False,
        dilate=0,
    )

    mask1_expected = np.array(
        [[0, 1, 0, 2], [0, 0, 0, 0], [3, 3, 3, 0], [3, 3, 3, 0]]
    )

    mask2_expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [255, 255, 255, 255],
            [255, 255, 255, 255],
            [255, 255, 255, 255],
        ]
    )

    mask3_expected = np.array(
        [[0, 255, 0, 255], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )

    assert np.all(mask1 == mask1_expected)
    assert np.all(mask2 == mask2_expected)
    assert np.all(mask3 == mask3_expected)
