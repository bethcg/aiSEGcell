#######################################################################################################################
# Sample data function for plugin.                                                                                    #
# © 2020-2024, ETH Zurich                                                                                             #
# Author:               Daniel Schirmacher, Timm Schroeder                                                            #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                               #
# Python Version:       3.8.7                                                                                         #
# PyTorch Version:      1.7.1                                                                                         #
# PyTorch Lightning Version: 1.5.9                                                                                    #
#######################################################################################################################

"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

import os

import pooch
from skimage import io


def make_sample_data():
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image

    # fetch image
    path_img = pooch.retrieve(
                url=(
                    'https://github.com/CSDGroup/aisegcell/raw/'
                    '521be0b66d497791d82e75c8211ac62cb31f6a2e/images/bf1.png'
                 ),
                known_hash=(
                        '06047311573206ee8c31716ac0e25116a18159'
                        'e48c1201c7d8b3cdbd398fc2b3'
                    ),
                fname='bf1.png',
                path=pooch.os_cache('napari_aisegcell'),
                progressbar=True
            )

    img = io.imread(path_img)

    return [(img, {"name": "bf1"})]
