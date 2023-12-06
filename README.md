# Wrapped $\beta$-Gaussians with compact support for exact probabilistic modeling on manifolds

## Implementation
Implementation of the Wrapped $\beta$-Gaussian distribution in PyTorch ([openreview](https://openreview.net/forum?id=KrequDpWzt)).
The `wbgauss` folder contains the implementation of the Wrapped $\beta$-Gaussian distribution:
- `wbgauss/sphere.py` contains the implementation of Wrapped $\beta$-Gaussian on the sphere with two parametrizations: pole and embedded;
- `wbgauss/special_orthogonal.py` contains the implementation of Wrapped $\beta$-Gaussian on the special orthogonal group.


## Installation
```bash
pip install -e .
```

## Experiments
`experiments/` folder contains code for the experiments from the paper. To run the experiments, you need to install the package first. Experiments can be run from the subfolders of `experiments/` folder.

## How to cite
```
@article{
  troshin2023wrapped,
  title={Wrapped \${\textbackslash}beta\$-Gaussians with compact support for exact probabilistic modeling on manifolds},
  author={Sergey Troshin and Vlad Niculae},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2023},
  url={https://openreview.net/forum?id=KrequDpWzt},
  note={}
}
```
