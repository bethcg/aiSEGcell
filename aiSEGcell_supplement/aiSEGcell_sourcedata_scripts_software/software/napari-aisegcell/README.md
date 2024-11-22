# napari-aiSEGcell

[![DOI](https://zenodo.org/badge/610379780.svg)](https://zenodo.org/doi/10.5281/zenodo.10091929)
[![License BSD-3](https://img.shields.io/pypi/l/napari-aisegcell.svg?color=green)](https://github.com/CSDGroup/napari-aisegcell/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-aisegcell.svg?color=green)](https://pypi.org/project/napari-aisegcell)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-aisegcell.svg?color=green)](https://python.org)
[![tests](https://github.com/CSDGroup/napari-aisegcell/workflows/tests/badge.svg)](https://github.com/CSDGroup/napari-aisegcell/actions)
[![codecov](https://codecov.io/gh/CSDGroup/napari-aisegcell/branch/main/graph/badge.svg)](https://codecov.io/gh/CSDGroup/napari-aisegcell)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-aisegcell)](https://napari-hub.org/plugins/napari-aisegcell)

A [napari] plugin to segment cell nuclei and whole cells in bright field microscopy images. `napari-aisegcell` uses
[aisegcell] for segmentation. This plugin can only be used to use already trained models for segmentation. If you
want to train a new model use [aisegcell]. Please cite [this paper](#citation) if you are using this plugin in your 
research.

![Screenshot](https://github.com/CSDGroup/napari-aisegcell/raw/main/images/napari-aisegcell_screenshot.png)

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Contents
  - [Installation](#installation)
    - [Command line](#command-line)
    - [One-click](#one-click)
  - [Data](#data)
  - [Documentation](#documentation)
    - [First launch](#first-launch)
    - [Launch napari]((#launch-napari))
    - [Layer mode](#layer-mode)
    - [Batch mode](#batch-mode)
    - [Trained models](#trained-models)
  - [Image annotation tools](#image-annotation-tools)
  - [Citation](#citation)
  - [Contributing](#contributing)
  - [License](#license)
  - [Issues](#issues)

## Installation
There are two ways to install `napari-aisegcell`: First, you can install `napari-aisegcell` from command line. Second, you
have `napari` already installed as a graphical user interface (GUI) and install `napari-aisegcell` from the GUI menu.

We recommend the [command line](#command-line) installation as it provides fine-grained control of the
installation to prevent conflicts with existing napari plugins. Use the [one-click](#one-click) installation if
you do not want to concern yourself with virtual environments or the command line. Just be aware that using the
[one-click](#one-click) installation may introduce conflicts with already installed plugins or new plugin installations
may disrupt this plugin.

`napari-aisegcell` was tested with
```bash
OS = macOS 12.6.3/ubuntu 22.10/windows 10
python = 3.8.6
torch = 1.10.2
torchvision = 0.11.3
pytorch-lightning = 1.5.9
```

### Command line
`napari` must be installed from command line to install `napari aisegcell` from command line. Installation requires a
command line application (e.g. `Terminal`) with [git] and [python] installed. If you do not have python installed
already, we recommend installing it using the [Anaconda distribution](https://www.anaconda.com/products/distribution).
If you operate on `Windows` we recommend using `Anaconda Powershell Prompt` as command line application.
An introductory tutorial on how to use `git` and GitHub can be found
[here](https://www.w3schools.com/git/default.asp?remote=github).

1) (Optional) If you already [installed napari](https://napari.org/stable/#installation) in a
[virtual environment](https://realpython.com/python-virtual-environments-a-primer/) you can skip this step.
However, you may want to install `napari-aisegcell` in a fresh environment to avoid conflicts with existing
plugins. Create a new virtual environment for `napari`. [Here](https://testdriven.io/blog/python-environments/) is a
list of different python virtual environment tools. Open your command line application and create a (e.g. `conda`)
virtual environment

    ```bash
    conda create -n napari python=3.8
    ```

2) Activate your virtual environment that has `napari` installed or you want to install `napari` to

    ```bash
    conda activate napari
    ```

3) (Optional) Install `napari`. Skip this step if you have `napari` already installed.

    ```bash
    pip install "napari[all]"
    ```

3) (Optional) If you use `Anaconda Powershell Prompt`, install `git` through `conda`

    ```bash
    conda install -c anaconda git
    ```

4) Install `napari-aisegcell`

    1) from [PyPI]

      ```bash
      pip install napari-aisegcell
      ```
    2) from GitHub (= latest development version)

      ```bash
      pip install git+https://github.com/CSDGroup/napari-aisegcell.git
      ```

With step 4) completed you have successfully installed `napari-aisegcell`. You can proceed with the
[documentation](#documentation) on how to use `napari-aisegcell`. *NOTE*, that when opening the plugin for the
first time, the remaining dependencies (`torch, torchvision, pytorch-lightning`) will be automatically installed
via [light-the-torch](https://github.com/pmeier/light-the-torch). If you prefer to manually install the remaining
dependencies (i.e. prevent potential interference with your virtual environment), proceed with step 5).

5) (Optional) `GPUs` greatly speed up training and inference of [aisegcell] and are available for `torch` (`v1.10.2`) 
for `Windows` and `Linux`. Check if your `GPU(s)` are CUDA compatible
([`Windows`](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/#verify-you-have-a-cuda-capable-gpu),
 [`Linux`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#verify-you-have-a-cuda-capable-gpu)) and
 update their drivers if necessary.

6) (Optional) [Install `torch`/`torchvision`](https://pytorch.org/get-started/previous-versions/) compatible with your
system. `aisegcell` was tested with `torch` version `1.10.2`, `torchvision` version `0.11.3`, and `cuda` version
`11.3.1`. Depending on your OS, your `CPU` or `GPU` (and `CUDA` version) the installation may change

```bash
# Windows/Linux CPU
pip install torch==1.10.2+cpu torchvision==0.11.3+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Windows/Linux GPU (CUDA 11.3.X)
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# macOS CPU
pip install torch==1.10.2 torchvision==0.11.3

```

7) (Optional) [Install `pytorch-lightning`](https://www.pytorchlightning.ai). `aisegcell` was tested with
version `1.5.9`.

```bash
# note the installation of v1.5.9 does not use pip install lightning
pip install pytorch-lightning==1.5.9
```

### One-click
(*NOT YET AVAILABLE*) Using the one-click installation of `napari-aisegcell` is as easy as opening `napari`, selecting
`Plugins>Install/Uninstall Plugins...` and searching for `napari-aisegcell` in the search bar. Select `install` and
restart `napari` for `napari-aisegcell` to appear in the list of installed plugins in the `Plugins` menu. Please
recall that one-click installing `napari-aisegcell` may interfere with existing plugin installations or new
plugin installations may interfere with the `napari-aisegcell` plugin.

## Data
`napari-aisegcell` is currently intended for single-class semantic and instance segmentation. Input images are expected 
to be 8-bit or 16-bit greyscale images. Segmentation masks are expected to decode background as 0 intensity and all intensities
\>0 are converted to a single intensity value (255). Have a look at
[this notebook](https://github.com/CSDGroup/aisegcell/blob/main/notebooks/data_example.ipynb)
for a data example.

## Documentation
`napari-aisegcell` has two modes: The [layer mode](#layer-mode) and the [batch mode](#batch-mode). Layer mode is
intended to explore if existing trained models in `napari-aisegcell` are suitable for your data. Batch mode is
intended for high-throughput image segmentation once you have confirmed that existing models are well suited for
your data.

### First launch
The first time you launch `napari-aisegcell`, `torch, torchvision, pytorch-lightning` will be automatically
installed if you have skipped steps 5)-7) of the [installation](#installation). The napari window will freeze
during download and installation. Depending on your setup this may take several minutes (~GBs of download). Similarly,
the first time you are running a pre-trained model the model weights (.CKPT file) will be downloaded and will
delay the prediction (~MBs of download).

### Launch napari
The `napari` bundled app is launched by opening the desktop shortcut. To launch napari as a command line
application

1) Open the terminal and activate the environment you installed `napari-aisegcell` into

    ```bash
    conda activate napari
    ```
2) Run `napari` in terminal

    ```bash
    napari
    ```

### Layer mode
Open the layer mode in the menu `Plugins>napari-aisegcell>Layer mode`. Select the parameters you want to use to
obtain your segmentation and select the `Run` button.

![Layer mode](https://github.com/CSDGroup/napari-aisegcell/raw/main/images/napari-aisegcell_layer_mode.png)

#### Data section
In the Data section (magenta) you can select the images you want to segment from a drop-down menu. Only images that are
loaded in the `napari` [layer tab](https://napari.org/stable/guides/layers.html) are available for selection.
Image formats must be readable by [`skimage.io.imread`](https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread).

#### Neural network section
In the neural network section (cyan) you can select the neural net you want to use for segmentation. The section
has three parameters

  - `Model type`
    - `NucSeg`: Select this option to use a pre-trained [aisegcell] model to segment nuclei (see [trained models](#trained-models)).
    - `CellSeg`: Select this option to use a pre-trained [aisegcell] model to segment whole cells (see [trained models](#trained-models)).
    - `Custom`: Select this option if you want to load a [aisegcell] model that does not ship with `napari-aisegcell`.
      Custom models can be obtained by training your own [aisegcell] model or obtaining [aisegcell] models from 3rd
      parties. You must select the checkpoint (.ckpt) file in the emerging `Custom Model` parameter.
  - `Pre-trained Model`: Drop-down menu to select the available pre-trained models for `NucSeg` or `CellSeg`.
    - `Bright Field`: a model to segment nuclei/whole cells in bright field. Currently, no other image modalities
      are available.
  - `Computing Device`: Drop-down menu that lists computing devices (CPU/GPUs) available with your current `torch`
    installation. 

#### Post-processing section
In the post-processing section (green) a selection of common post-processing steps are available

  - `Instance Segmentation`: Check this box to return instance segmentations instead of semantic segmentations.
  - `Remove Holes <`: Removes holes in objects (e.g. nuclei) \<X pixels.
  - `Minimum object size`: Removes objects of size \<X pixels before the dilation step.
  - `Maximum object size`: Removes objects of size \>X pixels before the dilation step.
  - `Dilation`: Dilate (\>0) or erode (\<0) objects by X pixels (see [here](https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html)).

### Batch mode
Open the batch mode in the menu `Plugins>napari-aisegcell>Batch mode`. Select the parameters you want to use to
obtain your segmentations and select the `Run` button.

The [neural network section](#neural-network-section) and [post-processing-section](#post-processing-section) are
the same as in the layer mode.

![Batch mode](https://github.com/CSDGroup/napari-aisegcell/raw/main/images/napari-aisegcell_batch_mode.png)

#### Data section
In the data section (magenta) you can select the images you want to submit for batch processing.

  - `Input`
    - `Select file`: Select an existing input CSV file that follows the [aisegcell input format](https://github.com/CSDGroup/aisegcell/blob/main/notebooks/unet_example.ipynb)
    - `Create file`: Create a CSV file that follows the [aisegcell input format](https://github.com/CSDGroup/aisegcell/blob/main/notebooks/unet_example.ipynb)
      - `Input directory`: Select parent directory containing all images to segment. Images can be in subdirectories.
      - `File pattern`: All files matching this pattern in your selected `Input directory` will be stored in the CSV file.
        Use [wildcard characters](https://linuxhint.com/bash_wildcard_tutorial/) like `*` to capture all images
        you want to segment in one run.
        - Example 1 `*/*.png`: will select all `PNG` files in in all sub-directories of `Input directory`.
        - Example 2 `position*z1.png`: will select all files in `Input directory` that start with "position" and
          end with "z1.png"
      - `Save as`: Name (and path) of the file that will be the input to your selected Neural Network. The file
        can be used as input to `Select file`.
      - `Mask suffix`: Mask suffix that will be appended to each mask file name.
        - Example `suffix = "_mask"`: `my_image.png` -> `my_image_mask.png`
      - `Output`:
        - `Directory`: Directory to which all segmentation masks will be saved. Be aware that input images with 
          identical file names will be appended with `Mask suffix` AND an ID. 
        - `CSD format`: Store segmentation masks following the storage system of the [Cell Systems Dynamics group](https://bsse.ethz.ch/csd).
          CSD format finds the deepest common directory of all input images, creates folders 'Analysis/Segmentation_YYMMDD', 
          and reconstructs the unique parts of all input paths in "Segmentation_YYMMDD".

### Trained models
We provide trained models:

| modality | image format | model | example image | description | availability |
| :-- | :-: | :-: | :-: | :-: | :-- |
| bright field nucleus segmentation | 2D grayscale | U-Net | <img src="https://github.com/CSDGroup/aisegcell/raw/main/images/nucseg.png" title="example nucleus segmentation" width="180px" align="center"> | Trained on a data set (link to data set) of 9849 images (~620k nuclei). | [ETH Research Collection](https://www.research-collection.ethz.ch/handle/20.500.11850/608641) |
| bright field whole cell segmentation | 2D grayscale | U-Net | <img src="https://github.com/CSDGroup/aisegcell/raw/main/images/cellseg.png" title="example whole cell segmentation" width="180px" align="center"> | Trained on a data set (link to data set) of 224 images (~12k cells). | [ETH Research Collection](https://www.research-collection.ethz.ch/handle/20.500.11850/608646) |


## Image annotation tools
Available tools to annotate segmentations include:

  - [napari](https://napari.org/stable/)
  - [Labkit](https://imagej.net/plugins/labkit/) for [Fiji](https://imagej.net/software/fiji/downloads)
  - [QuPath](https://qupath.github.io)
  - [ilastik](https://www.ilastik.org)

## Citation
t.b.d.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-aisegcell" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/CSDGroup/napari-aisegcell/issues

[aisegcell]: https://github.com/CSDGroup/aisegcell
[git]: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
[tox]: https://tox.readthedocs.io/en/latest/
[PyPI]: https://pypi.org/
[python]: https://www.python.org
