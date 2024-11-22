#######################################################################################################################
# Batch mode widget.                                                                                                  #
# © 2020-2024, ETH Zurich                                                                                             #
# Author:               Daniel Schirmacher, Timm Schroeder                                                            #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                               #
# Python Version:       3.8.7                                                                                         #
# PyTorch Version:      1.7.1                                                                                         #
# PyTorch Lightning Version: 1.5.9                                                                                    #
#######################################################################################################################

"""
TODO:
* output: directory on server
    * make server ssh accessible with openssh
    (https://learn.microsoft.com/en-us/windows-server/administration/
    openssh/openssh_install_firstuse?tabs=gui)
* device selection
    * Euler Cluster
        * use ssh to submit remotely: ssh -T account@euler.ethz.ch <<ENDSSH
            *some code* ENDSSH
        * send files from list to $SCRATCH
        * generate new file on Euler with $SCRATCH paths of images
        * save predictions on $SCRATCH
        * send back files to output directory
* run button: calls aisegcell_predict
    * euler cluster: needs to return JOBID to napari plugin for "cancel" and
    "status queury"
* status queury:
    * returns if job is in queue/running (how long)/estimate of when it will
    be finished?
* cancel button:
    * local: stop `aisegcell_predict` (how?)
    * euler cluster: send bkill command to cluster
* output has estimate of segmentation quality (GA's suggestion)
    * with GT mask (= IoU), without mask (predict IoU based on mask features)

"""

import os
from typing import TYPE_CHECKING, Union

from importlib.util import find_spec
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info

if find_spec("torch") is None:
    show_info("Please wait while torch is installed...")
    os.system("ltt install torch==1.10.2")

if find_spec("torchvision") is None:
    show_info("Please wait while torchvision is installed...")
    os.system("ltt install torchvision==0.11.3")

if find_spec("pytorch_lightning") is None:
    show_info("Please wait while pytorch-lightning is installed...")
    os.system("ltt install pytorch-lightning==1.5.9")

import torch

from magicgui import magicgui
from skimage import io

from napari_aisegcell._utils import (
    _postprocess,
    _preprocess,
    change_handler,
    check_order,
    rename_duplicates,
)

if TYPE_CHECKING:
    import napari

input_fmt_choices = [
    ("Select file", "select_file"),
    ("Create file", "create_file"),
]

file_output_def = os.path.join(
    os.path.join(os.path.expanduser("~")), "Desktop/input_files.csv"
)

output_fmt_choices = [
    ("Directory", "dir"),
    ("CSD format", "csd"),
]

model_type_choices = [
    ("NucSeg", "nucleus_segmentation"),
    ("CellSeg", "cell_segmentation"),
    ("Custom", "custom model"),
]

models_reg = {
    "model_nucseg": [("Bright Field", "bright_field")],
    "model_cellseg": [("Bright Field", "bright_field")],
}

device_choices = [("CPU", "cpu")]

if torch.cuda.is_available():
    device_choices += [
        (f"GPU {i} ({torch.cuda.get_device_name(i)})", f"cuda:{i}")
        for i in range(torch.cuda.device_count())
    ]

# -----------------------------------------------------------------------------

DEBUG = False

DEFAULTS = {
    "input_fmt": input_fmt_choices[0][1],
    "file_output": file_output_def,
    "suffix": "_mask",
    "output_fmt": output_fmt_choices[1][1],
    "model_type": model_type_choices[0][1],
    "model_nucseg": models_reg["model_nucseg"][0][1],
    "model_cellseg": models_reg["model_cellseg"][0][1],
    "device": device_choices[0][1],
    "instance_segmentation": False,
    "remove_holes": 1,
    "size_min": 1,
    "size_max": 1000000,
    "dilation": 0,
}

TOOLTIPS = {
    "input_fmt": "Select an existing input file or create a new one.",
    "file_select": (
        "Select a CSV file containing paths to image to \n"
        "segment in the first column. File requires a header."
    ),
    "dir_input": (
        "Select parent directory containing all images to segment"
        ". \nImages can be in subdirectories."
    ),
    "file_pattern": (
        "All files matching this pattern in your selected \n"
        "'Input directory' will be stored in the CSV file. Images can \n"
        "be in subdirectories."
    ),
    "file_output": (
        "Select CSV file to store input paths to. Example: \n"
        "'~/Desktop/input_paths.csv'"
    ),
    "output_fmt": (
        "Store all masks in a single directory or in Cell \n"
        "Systems Dynamics (CSD) group format. CSD format \n"
        "finds the deepest common directory of all input images, \n"
        "creates folders 'Analysis/Segmentation_YYMMDD', and \n"
        "reconstructs the unique parts of all input paths in \n"
        "'Segmentation_YYMMDD'."
    ),
    "dir_output": "Select directory to store masks in.",
    "suffix": "Select suffix that will be appended to each mask file name.",
    "model_nucseg": "Select mode of pretrained model.",
    "model_cellseg": "Select mode of pretrained model.",
    "model_custom": "Select path to custom model.",
    "device": "Computing device used for predictions.",
    "instance_segmentation": "Check to enable instance segmetation.",
    "remove_holes": "Remove holes in objects of size < X [px].",
    "size_min": "Remove objects of size < X [px] prior to dilation.",
    "size_max": "Remove objects of size > X [px] prior to dilation.",
    "dilation": "Dilate (>0) or shrink (<0) mask by X [px].",
}

# -----------------------------------------------------------------------------


def make_batch_mode_widget():
    @magicgui(
        label_data=dict(widget_type="Label", label="<br><b>Data:</b>"),
        input_fmt=dict(
            widget_type="RadioButtons",
            label="Input",
            orientation="horizontal",
            choices=input_fmt_choices,
            value=DEFAULTS["input_fmt"],
            tooltip=TOOLTIPS["input_fmt"],
        ),
        file_select=dict(
            widget_type="FileEdit",
            visible=False,
            label="Input file",
            tooltip=TOOLTIPS["file_select"],
            mode="r",
            filter="*.csv",
        ),
        dir_input=dict(
            widget_type="FileEdit",
            visible=False,
            label="Input directory",
            tooltip=TOOLTIPS["dir_input"],
            mode="d",
        ),
        file_pattern=dict(
            widget_type="LineEdit",
            visible=False,
            label="File pattern",
            tooltip=TOOLTIPS["file_pattern"],
        ),
        file_output=dict(
            widget_type="FileEdit",
            visible=False,
            label="Save as",
            tooltip=TOOLTIPS["file_output"],
            mode="w",
            filter="*.csv",
            value=DEFAULTS["file_output"],
        ),
        suffix=dict(
            widget_type="LineEdit",
            visible=False,
            label="Mask suffix",
            tooltip=TOOLTIPS["suffix"],
            value=DEFAULTS["suffix"],
        ),
        output_fmt=dict(
            widget_type="RadioButtons",
            visible=False,
            label="Output",
            orientation="horizontal",
            choices=output_fmt_choices,
            value=DEFAULTS["output_fmt"],
            tooltip=TOOLTIPS["output_fmt"],
        ),
        dir_output=dict(
            widget_type="FileEdit",
            visible=False,
            label="Output directory",
            tooltip=TOOLTIPS["dir_output"],
            mode="d",
        ),
        label_nn=dict(
            widget_type="Label", label="<br><b>Neural Network Selection:</b>"
        ),
        model_type=dict(
            widget_type="RadioButtons",
            label="Model Type",
            orientation="horizontal",
            choices=model_type_choices,
            value=DEFAULTS["model_type"],
        ),
        model_nucseg=dict(
            widget_type="ComboBox",
            visible=False,
            label="Pre-trained Model",
            tooltip=TOOLTIPS["model_nucseg"],
            choices=models_reg["model_nucseg"],
            value=DEFAULTS["model_nucseg"],
        ),
        model_cellseg=dict(
            widget_type="ComboBox",
            visible=False,
            label="Pre-trained Model",
            tooltip=TOOLTIPS["model_cellseg"],
            choices=models_reg["model_cellseg"],
            value=DEFAULTS["model_cellseg"],
        ),
        model_custom=dict(
            widget_type="FileEdit",
            visible=False,
            label="Custom Model",
            tooltip=TOOLTIPS["model_custom"],
            mode="r",
        ),
        device=dict(
            widget_type="ComboBox",
            visible=True,
            label="Computing Device",
            tooltip=TOOLTIPS["device"],
            choices=device_choices,
            value=DEFAULTS["device"],
        ),
        label_postprocessing=dict(
            widget_type="Label", label="<br><b>Post-processing:</b>"
        ),
        instance_segmentation=dict(
            widget_type="CheckBox",
            label="Instance Segmentation",
            tooltip=TOOLTIPS["instance_segmentation"],
            value=DEFAULTS["instance_segmentation"],
        ),
        remove_holes=dict(
            widget_type="SpinBox",
            label="Remove Holes <",
            tooltip=TOOLTIPS["remove_holes"],
            value=DEFAULTS["remove_holes"],
            min=1,
            max=1000000,
            step=1,
        ),
        size_min=dict(
            widget_type="SpinBox",
            label="Minimum object size",
            tooltip=TOOLTIPS["size_min"],
            value=DEFAULTS["size_min"],
            min=1,
            max=1000000,
            step=1,
        ),
        size_max=dict(
            widget_type="SpinBox",
            label="Maximum object size",
            tooltip=TOOLTIPS["size_max"],
            value=DEFAULTS["size_max"],
            min=1,
            max=1000000,
            step=1,
        ),
        dilate=dict(
            widget_type="SpinBox",
            label="Dilation",
            tooltip=TOOLTIPS["dilation"],
            value=DEFAULTS["dilation"],
            min=-1000,
            max=1000,
            step=1,
        ),
    )
    def batch_mode_widget(
        viewer: "napari.viewer.Viewer",
        label_data: str,
        input_fmt: str,
        file_select: str,
        dir_input: str,
        file_pattern: str,
        file_output: str,
        suffix: str,
        output_fmt: str,
        dir_output: str,
        label_nn: str,
        model_type: str,
        model_nucseg: str,
        model_cellseg: str,
        model_custom: str,
        device: str,
        label_postprocessing: str,
        instance_segmentation: bool,
        remove_holes: int,
        size_min: int,
        size_max: int,
        dilate: int,
    ) -> None:
        # import packages at run time
        import glob
        from datetime import date

        import numpy as np
        import pandas as pd
        import pooch
        import torch
        from aisegcell.models.unet import LitUnet

        # Fetch models
        if model_type == "nucleus_segmentation":
            path_model = pooch.retrieve(
                        url=(
                            'https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/608641/'
                            'best-f1-epoch377-step239651.ckpt?sequence=2&isAllowed=y'
                         ),
                        known_hash='7e302470af7e2aba5bd456082a6185aa73417eff49330554f9cd6382264f9b1f',
                        fname='nucseg_model.ckpt',
                        path=pooch.os_cache('napari_aisegcell'),
                        progressbar=True
                    )
        elif model_type == "cell_segmentation":
            path_model = pooch.retrieve(
                        url=(
                            'https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/608646/'
                            'best-f1-epoch345-step9341.ckpt?sequence=1&isAllowed=y'
                         ),
                        known_hash='6c15e7ea7d8b035f7793b9a68bbce7819c5189a0815ac24bd5164201f127379f',
                        fname='cellseg_model.ckpt',
                        path=pooch.os_cache('napari_aisegcell'),
                        progressbar=True
                    )
        elif model_type == "custom model":
            path_model = model_custom

        # set up input file
        if input_fmt == "select_file":
            assert (
                ".csv" in file_select.as_posix()
            ), "Input file must be a CSV file."
            path_input = file_select
        elif input_fmt == "create_file":
            path_input = file_output
            files = sorted(glob.glob(os.path.join(dir_input, file_pattern)))
            assert (
                len(files) > 0
            ), f'No files with pattern "{file_pattern}" at "{dir_input}".'

            tmp = pd.DataFrame({"bf": files})
            tmp.to_csv(path_input, index=False)
            del tmp, files
        else:
            raise ValueError(
                "input_fmt must be in (select_file, "
                f"create_file), but is {input_fmt}"
            )

        # set up output formats
        tmp = pd.read_csv(path_input)
        if input_fmt == "select_file":
            assert "out" in tmp.columns, 'file_select is missing column "out".'
            dirs = [
                os.sep.join(d.split(os.sep)[:-1]) for d in tmp.out.tolist()
            ]
            output_base_path = os.path.commonpath(dirs)

            for d in np.unique(dirs):
                os.makedirs(d, exist_ok=True)

            del dirs
        else:
            if output_fmt == "dir":
                output_base_path = dir_output

                # construct output paths
                files = [f.split(os.path.sep)[-1] for f in tmp.bf.tolist()]

                assert check_order(
                    tmp.bf.tolist(), files
                ), "Order of input images and output paths does not match."

                files = [
                    ".".join([f.split(".")[-2] + suffix, f.split(".")[-1]])
                    for f in files
                ]
                files = [os.path.join(output_base_path, f) for f in files]

                # rename duplicate filenames
                files = rename_duplicates(files)

                tmp.loc[:, "out"] = files

                tmp.to_csv(path_input, index=False)
                del files
            elif output_fmt == "csd":
                # find maximum common path
                if len(tmp.bf.tolist()) > 1:
                    output_base_path = os.path.commonpath(tmp.bf.tolist())
                else:
                    output_base_path = os.path.dirname(tmp.bf.tolist()[0])

                # change filepaths to relative paths
                files = [
                    os.path.relpath(f, output_base_path)
                    for f in tmp.bf.tolist()
                ]
                files = [
                    ".".join([f.split(".")[-2] + suffix, f.split(".")[-1]])
                    for f in files
                ]

                # add segmentation folder to path
                today = date.today().strftime("%y%m%d")
                output_base_path = os.path.join(
                    output_base_path, "Analysis", f"Segmentation_{today}"
                )

                i = 1
                while os.path.exists(output_base_path):
                    output_base_path += f"_{i}"
                    i += 1

                files = [os.path.join(output_base_path, f) for f in files]

                # rename duplicate filenames
                files = rename_duplicates(files)

                # create all sub-directories
                dirs = [
                    os.path.sep.join(f.split(os.path.sep)[:-1]) for f in files
                ]

                for d in np.unique(dirs):
                    os.makedirs(d, exist_ok=True)

                # save all output paths to path_input
                tmp.loc[:, "out"] = files
                tmp.to_csv(path_input, index=False)

                del i, d, today, dirs, files
            else:
                raise ValueError(
                    f"output_fmt must be in (dir, csd), but is '{output_fmt}'."
                )

        # start prediction loop
        @thread_worker(
            progress={"total": len(tmp), "desc": "segmentation progress"}
        )
        def predict_wrapper(data: pd.DataFrame, path_model: str, device: str):
            for i in range(len(data)):
                # load image
                img = io.imread(data.bf.iloc[i], plugin="pil")
                img_t = _preprocess(img=img, device=device)

                # load model checkpoint for prediction
                model = LitUnet.load_from_checkpoint(path_model)
                model = model.to(device)

                model.eval()

                # obtain mask
                with torch.no_grad():
                    mask = model(img_t)

                mask[mask < 0.5] = 0
                mask[mask >= 0.5] = 1

                # convert prediction to numpy array for post-processing
                mask = (
                    mask.mul(255)
                    .add_(0.5)
                    .clamp_(0, 255)
                    .to("cpu", torch.uint8)
                    .numpy()[0, 0, :, :]
                )

                mask = _postprocess(
                    mask=mask,
                    remove_holes=remove_holes,
                    size_min=size_min,
                    size_max=size_max,
                    instance_segmentation=instance_segmentation,
                    dilate=dilate,
                )

                if not instance_segmentation:
                    mask = mask.astype(np.uint8)

                io.imsave(data.out.iloc[i], mask)

                yield i

        worker = predict_wrapper(
            data=tmp, path_model=path_model, device=device
        )
        viewer.window._status_bar._toggle_activity_dock(True)
        worker.start()

        # button = QPushButton("STOP!")
        # button.clicked.connect(worker_predict.quit)
        # button.clicked.connect(worker_progress.quit)
        # worker_predict.finished.connect(button.clicked.disconnect)
        # worker_progress.finished.connect(button.clicked.disconnect)
        # viewer.window.add_dock_widget(button)

        # predict_wrapper
        #   1. create postprocessing wrapper
        #   2. instantiate worker_postprocessing
        #   3. if worker_predict yields send output worker_postprocessing
        #      and resume
        #   4. on worker_postprocessing yielded call
        #      worker_postprocessing.pause

        #   works if computing on local GPU
        #   does not work when computing on cluster -> use aisegcell_predict

        # TODO: cluster mode for submissions
        #   submit job with scp and provide email alert when the job is done
        #   job is submitted by pressing run plugin is not blocked anymore

    # widgets for input_fmt
    widget_for_input_fmt = {
        "select_file": [batch_mode_widget.file_select],
        "create_file": [
            batch_mode_widget.dir_input,
            batch_mode_widget.file_pattern,
            batch_mode_widget.file_output,
            batch_mode_widget.suffix,
            batch_mode_widget.output_fmt,
        ],
    }

    # widgets for model types
    widget_for_modeltype = {
        "nucleus_segmentation": batch_mode_widget.model_nucseg,
        "cell_segmentation": batch_mode_widget.model_cellseg,
        "custom model": batch_mode_widget.model_custom,
    }

    # input_fmt widget triggers a change event initially
    @change_handler(batch_mode_widget.input_fmt, init=False, debug=DEBUG)
    def _input_fmt_change(input_fmt: str):
        selected = widget_for_input_fmt[input_fmt]
        for w in {
            batch_mode_widget.file_select,
            batch_mode_widget.dir_input,
            batch_mode_widget.file_pattern,
            batch_mode_widget.file_output,
            batch_mode_widget.suffix,
            batch_mode_widget.output_fmt,
        } - {x for x in selected}:
            w.hide()

        for s in selected:
            s.show()
            s.changed(s.value)

        if input_fmt == "select_file":
            batch_mode_widget.output_fmt.value = "csd"

    # output_fmt widget triggers a change event initially
    @change_handler(batch_mode_widget.output_fmt, init=False, debug=DEBUG)
    def _output_fmt_change(output_fmt: str):
        if output_fmt == "dir":
            batch_mode_widget.dir_output.show()
            batch_mode_widget.dir_output.changed(
                batch_mode_widget.dir_output.value
            )
        else:
            batch_mode_widget.dir_output.hide()

    # model_type widget triggers a change event initially
    @change_handler(batch_mode_widget.model_type, init=False, debug=DEBUG)
    def _model_type_change(model_type: Union[str, type]):
        selected = widget_for_modeltype[model_type]
        for w in {
            batch_mode_widget.model_nucseg,
            batch_mode_widget.model_cellseg,
            batch_mode_widget.model_custom,
        } - {selected}:
            w.hide()
        selected.show()

        # trigger _model_change
        selected.changed(selected.value)

    return batch_mode_widget
