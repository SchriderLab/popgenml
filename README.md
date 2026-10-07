# popgenml

A repo with tools to simulate population genetic scenarios, apply popular inference routines such as Relate and SINGER, and to train machine learning inference models all from within Python. 


Includes support for popular popgen simulators:

* [msprime](https://github.com/tskit-dev/msprime)
* [SLiM](https://github.com/MesserLab/SLiM)
* [discoal](https://github.com/kr-colab/discoal)

Python wrappings for popular inference routines:

* [Relate](https://myersgroup.github.io/relate/index.html) - tree sequence inference (<https://myersgroup.github.io/relate/index.html>)
* [SINGER](https://github.com/popgenmethods/SINGER) - tree sequence inference

Formatting routines:

* Computation of windowed population genetic statistics (such as LD or the site frequency spectrum)
* Seriation
* Linear sum assignment for subpopulations
* FW encoding of inferred or ground truth genealogical trees (https://www.pnas.org/doi/10.1073/pnas.1922851117)
* Conversions between TSKit trees and distance matrices and graphs (node and edge sets)

Inference models:

* ResNet for inference on genotype matrices
* UNet from (https://pmc.ncbi.nlm.nih.gov/articles/PMC9979274/)
* GCN models from (https://academic.oup.com/mbe/article/41/11/msae223/7845315)

See the tutorials folder for examples as Jupyter notebooks.  The documentation and API reference can be found at https://popgenml.readthedocs.io/en/latest/

## Installation

### 1. Prerequisites: External Simulators and Tools

To run simulations and inference routines utilizing SLiM, Relate, and SINGER, you must have their C++ binaries installed. We provide an automated installation script that compiles and installs these tools to your local user environment (`~/.local/bin`) without requiring root access.

First, ensure you have standard C++ build tools installed on your system. For Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake git zlib1g-dev
```

Then, run the provided companion installation script from the root of this repository:

```bash
chmod +x install_popgen_tools.sh
./install_popgen_tools.sh
```

*Note: Ensure `~/.local/bin` is in your system's `$PATH`. The script will warn you if you need to add it to your `~/.bashrc` or `~/.zshrc`.*

### 2. Torch and torch-geometric (conda)

First, create a fresh environment with Python 3.10:

```bash
conda create -n "popgenml" python=3.10
conda activate popgenml
```

**Install PyTorch with CUDA support**

To utilize GPU acceleration, ensure your PyTorch installation matches your system's CUDA version. You can check your available CUDA driver version by running `nvidia-smi`.

For the latest PyTorch distributions with CUDA 12.x (e.g., CUDA 12.4), use the official Conda channels:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
*(If you do not have a GPU, you can install the CPU-only version by omitting the `pytorch-cuda` flag and using `cpuonly` instead).*

**Install PyTorch Geometric (PyG)**

You can install the core PyTorch Geometric library directly via conda:

```bash
conda install pyg -c pyg
```

*Optional but highly recommended for performance:* PyG relies on a few C++/CUDA extension packages (like `torch_scatter` and `torch_sparse`) for efficient graph operations. To avoid compiling these from source, PyG hosts pre-compiled pip wheels that must strictly match your PyTorch (`${TORCH}`) and CUDA (`${CUDA}`) versions.

For example, if you installed PyTorch 2.5 and CUDA 12.4, install the extensions via pip like this:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

### 3. Install popgenml

Finally, you can install the `popgenml` python package:

```bash
git clone https://github.com/SchriderLab/popgenml/
cd popgenml
pip install -r requirements.txt
python setup.py install
```