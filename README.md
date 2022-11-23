# Effective Margin Regularization
Code for "[Boosting Adversarial Robustness From The Perspective of Effective Margin Regularization](https://arxiv.org/abs/2210.05118)", BMVC 2022

## Requirement
PyTorch >= 1.9.0

## Data preparation
We use the default CIFAR10 of Pytorch.


## Train

Use the bash script in this repository. You can change the arguments in the bash file, such as data path, hyperparameters and result file path.

AT-EMR:
```
bash train-AT-EMR.sh
```

TRADES-EMR:
```
bash train-TRADES-EMR.sh
```
For the about experiment, we split the trainin set into two subsets with 48,000 and 2000 images respectively. The 2000 images are for validation and model selection (early-stopping) to avoid robust overfitting.

The following experiment is adversarial training with [MAIL loss](https://github.com/QizhouWang/MAIL).

AT-MAIL with EMR:
```
bash train-MAIL-AT-EMR.sh
```

TRADES-MAIL with EMR:
```
bash train-MAIL-TRADES-EMR.sh
```

### Citation
If you use our code in your research, please cite with:

```
@InProceedings{liu2022boosting,
    author    = {Liu, Ziquan and Chan, Antoni B.},
    title     = {Boosting Adversarial Robustness From The Perspective of Effective Margin Regularization},
    booktitle = {British Machine Vision Conference (BMVC)},
    year      = {2022},
}
```

### Acknowledgement
We use [MAIL loss](https://github.com/QizhouWang/MAIL) package in the MAIL loss computation.
