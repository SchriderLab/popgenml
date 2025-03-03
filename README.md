# popgenml

This repo and its docs are a work in progress.  

A repo with tools to simulate population genetic scenarios, apply popular inference routines such as Relate, and to train machine learning inference models all from within Python. 

Includes support for popular popgen simulators:

* msprime (https://github.com/tskit-dev/msprime)
* SLiM (https://github.com/MesserLab/SLiM)

Includes inference routines:

* Relate - tree sequence inference (https://myersgroup.github.io/relate/index.html)

Formatting routines:

* Seriation
* Linear sum assignment for subpopulations
* FW encoding of inferred or ground truth genealogical trees (https://www.pnas.org/doi/10.1073/pnas.1922851117)

Inference models:

* ResNet for inference on genotype matrices
* UNet from (https://pmc.ncbi.nlm.nih.gov/articles/PMC9979274/)
* GCN models from (https://academic.oup.com/mbe/article/41/11/msae223/7845315)

## Installation

### Prerequisites

Install SLiM on Linux:

```
git clone https://github.com/MesserLab/SLiM.git
cd SLiM
mkdir build
cd build
cmake ..
make
sudo make install
```

Or on MacOS you can use the Messer Lab's installer (https://messerlab.org/slim/).

### Torch and torch-geometric (conda)

```
conda create -n "popgenml" python=3.9
# install torch
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# install pytorch-geometric
conda install pyg -c pyg
```

Finally you can install the python package like:

```
git clone https://github.com/SchriderLab/popgenml/
cd popgenml
pip install -r requirements.txt
python3 setup.py install
```

