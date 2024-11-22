#######################################################################################################################
# Tests of Layer mode.                                                                                                #
# © 2020-2024, ETH Zurich                                                                                             #
# Author:               Daniel Schirmacher, Timm Schroeder                                                            #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                               #
# Python Version:       3.8.7                                                                                         #
# PyTorch Version:      1.7.1                                                                                         #
# PyTorch Lightning Version: 1.5.9                                                                                    #
#######################################################################################################################

from typing import Callable

import napari
import pytest
from skimage.measure import label


@pytest.fixture
def viewer_widget(make_napari_viewer: Callable[..., napari.Viewer]):
    # from https://github.com/MouseLand/cellpose-napari
    viewer = make_napari_viewer()
    _, widget = viewer.window.add_plugin_dock_widget(
        plugin_name="napari-aisegcell", widget_name="Layer mode"
    )
    return viewer, widget


def test_basic_function(qtbot, viewer_widget):
    viewer, widget = viewer_widget
    assert len(viewer.window._dock_widgets) == 1

    viewer.open_sample("napari-aisegcell", "bf1")
    viewer.layers[0].data = viewer.layers[0].data[-256:, -256:]

    # repeated model download yields HTTP error 429
    # widget()  # run segmentation with all default parameters

    # check widget attributes
    assert hasattr(widget, "img")
    assert hasattr(widget, "label_nn")
    assert hasattr(widget, "model_type")
    assert hasattr(widget, "model_nucseg")
    assert hasattr(widget, "model_cellseg")
    assert hasattr(widget, "model_custom")
    assert hasattr(widget, "device")
    assert hasattr(widget, "label_postprocessing")
    assert hasattr(widget, "instance_segmentation")
    assert hasattr(widget, "remove_holes")
    assert hasattr(widget, "size_min")
    assert hasattr(widget, "size_max")
    assert hasattr(widget, "dilate")

    # check attribute types
    assert widget.img.annotation == napari.layers.image.image.Image
    assert widget.label_nn.annotation == str
    assert widget.model_type.annotation == str
    assert widget.model_nucseg.annotation == str
    assert widget.model_cellseg.annotation == str
    assert widget.model_custom.annotation == str
    assert widget.device.annotation == str
    assert widget.label_postprocessing.annotation == str
    assert widget.instance_segmentation.annotation == bool
    assert widget.remove_holes.annotation == int
    assert widget.size_min.annotation == int
    assert widget.size_max.annotation == int
    assert widget.dilate.annotation == int

    # check that the layers were created properly
    # assert len(viewer.layers) == 2
    # assert "mask_nucleus" in viewer.layers[-1].name

    # check that the segmentation was proper, should yield 2 cells
    # mask_inst = label(viewer.layers[-1].data)
    # assert mask_inst.max() == 2
