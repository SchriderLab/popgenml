# popgenml

A repo with tools to simulate population genetic scenarios, apply popular inference routines such as Relate and SINGER, and to train machine learning inference models all from within Python. 


Includes support for popular popgen simulators:

* [msprime](https://github.com/tskit-dev/msprime)
* [SLiM](https://github.com/MesserLab/SLiM)
* [discoal](https://github.com/kr-colab/discoal)

Python wrappings for popular tree sequence inference routines:

* [Relate](https://myersgroup.github.io/relate/index.html) 
* [SINGER](https://github.com/popgenmethods/SINGER) 

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

See the [tutorials](https://github.com/SchriderLab/popgenml/tree/main/tutorials/) folder for examples as Jupyter notebooks.  The documentation and API reference can be found at https://popgenml.readthedocs.io/en/latest/

## Installation

### Linux

To run simulations and inference routines utilizing SLiM, Relate, and SINGER, you must have their C++ binaries installed. We provide an automated installation script that compiles and installs these tools to your local user environment (`~/.local/bin`).

We recommend installing through Anaconda (https://www.anaconda.com/docs/getting-started/anaconda/install/linux-install).

First, ensure you have standard C++ build tools installed on your system. For Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake git zlib1g-dev
```

Then, create a fresh environment with Python 3.10 or greater and run the provided installation script from the root of this repo:

```bash
conda create -n "popgenml" python=3.10
conda activate popgenml

# make sure to run the shell script from within your conda environment
chmod +x setup.sh
./setup.sh
```

