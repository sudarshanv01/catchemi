# `catchemi`

Python package for performing calculations incorporating the Newns-Anderson model of chemisorption with the different orthogonalization terms. Current orthogonalization schemes include the Hammer-Nørksov *d*-band model 2-state term and the Newns-Anderson-Grimley model.

See [here](https://aip.scitation.org/doi/full/10.1063/5.0096625) for more details about the implementation.

## Installation

This package can be installed using a combination of `conda` and `pip`. If you use an M1 Mac, please see the special instructions below before starting the `conda` and `pip` dependecies.

Install conda depdencies:

```bash
conda env create -f environment.yml
conda activate catchemi
```

Install pip dependencies:

```bash
pip install -r requirements.txt
```

Install the package:

```bash
pip install .
```

(Optional) Install requirements for testing

```bash
pip install -r requirements-test.txt
```

(Optional) Install requirements for documentation

```bash
pip install -r requirements-docs.txt
```


### M1 Mac special instructions

If you are using an M1 Mac, you will need to install `flint` through `brew` before installing any of the above dependencies. 

```bash
brew install flint
```