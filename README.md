# Acquiring Diverse Skills using Curriculum Reinforcement Learning with Mixture of Experts

## Installation 
Create a conda environment with **python3.8**, activate it and install the packages listed below in the right order.

1. The Trust-Region Projection Layers use a C++ implementation for solving the dual. This is implemented in the  [ITPAL](https://github.com/ALRhub/ITPAL) (MIT LICENSE) package and needs to be seperately installed. Please follow the installation instructions listed [here](https://github.com/ALRhub/ITPAL).   

2. Install fancy gym--the package that contains the environments--(MIT LICENSE). Navigate to the fancy gym folder and install via pip
```bash
pip install -e .
```

Note that [Fancy gym](https://github.com/ALRhub/fancy_gym) was updated and has new version releases ensuring compatibility to new gymnasium versions, 
integration to other benchmark suits like DMC and metaworld and easy installation via pip. However, this project is based on an older version, therefore we have a built-in package in this repository.  

3. Di-Skill uses motion primitives. We use the [mp-pytorch](https://github.com/ALRhub/MP_PyTorch) package (GPL-3.0 LICENSE). We provide the used version in this repository. Navigate to MP_Pytorch and install the package via
```bash
pip install -e .
```

4. Additionally, we use [ClusterWorks2](https://github.com/ALRhub/cw2) (MIT LICENSE) for managing the experiments and potentially running scripts on an HPC. Navigate to the cw2 folder and install it via
```bash
pip install -e .
```

5. Finally, we can install all other dependencies needed for Di-Skill. Please note that Di-Skill partly uses code from the [Trust Region Projection Layers](https://github.com/boschresearch/trust-region-layers/tree/main?tab=readme-ov-file) which is under the AGPL-3.0 LICENSE. 
Hence, DI-Skill follows the AGPL-3.0 LICENSE. Navigate to the folder diskill and install it via
```bash
pip install -r requirements.txt
pip install -e .
```
## Running an experiment
```python
python diskill/run.py diskill/configs/table_tennis.yml -o --nocodecopy 
```
