# CS 182 Final Project
Members: Kenny, Catherine, Jennifer, Jaewon, Phillip

# Problem Statement
We will be exploring In-Context Learning (ICL) discussed in the paper ["What Can Transformers Learn In-Context? A Case Study of Simple Function Classes"](https://arxiv.org/abs/2208.01066) by Garg, et. al. Our advancements will be using more modern neural network architectures to learn a new class of problems: kernelized linear regression.

# Environment Setup
Installing the necessary packages
```
conda env create -f environment.yml
```

Setting up the jupyter notebook kernel for `eval.ipynb`
```
python -m ipykernel install --user --name=in-context-learning
```