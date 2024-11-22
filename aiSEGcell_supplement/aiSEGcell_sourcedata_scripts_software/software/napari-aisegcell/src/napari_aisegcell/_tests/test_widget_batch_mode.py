#######################################################################################################################
# Tests of Batch mode.                                                                                                #
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


@pytest.fixture
def viewer_widget(make_napari_viewer: Callable[..., napari.Viewer]):
    # from https://github.com/MouseLand/cellpose-napari
    viewer = make_napari_viewer()
    _, widget = viewer.window.add_plugin_dock_widget(
        plugin_name="napari-aisegcell", widget_name="Batch mode"
    )
    return viewer, widget


def test_basic_function(qtbot, viewer_widget):
    viewer, widget = viewer_widget
    assert len(viewer.window._dock_widgets) == 1

    # check widget attributes
    assert hasattr(widget, "label_data")
    assert hasattr(widget, "input_fmt")
    assert hasattr(widget, "file_select")
    assert hasattr(widget, "dir_input")
    assert hasattr(widget, "file_pattern")
    assert hasattr(widget, "file_output")
    assert hasattr(widget, "suffix")
    assert hasattr(widget, "output_fmt")
    assert hasattr(widget, "dir_output")
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
    assert widget.label_data.annotation == str
    assert widget.input_fmt.annotation == str
    assert widget.file_select.annotation == str
    assert widget.dir_input.annotation == str
    assert widget.file_pattern.annotation == str
    assert widget.file_output.annotation == str
    assert widget.suffix.annotation == str
    assert widget.output_fmt.annotation == str
    assert widget.dir_output.annotation == str
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
