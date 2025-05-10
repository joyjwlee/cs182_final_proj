# CS 182 Final Project
Members: Kenny, Catherine, Jennifer, Jaewon, Phillip

# Problem Statement
We will be exploring In-Context Learning (ICL) discussed in the paper ["What Can Transformers Learn In-Context? A Case Study of Simple Function Classes"](https://arxiv.org/abs/2208.01066) by Garg, et. al. Our advancements will be using more modern neural network architectures to learn a new class of problems: kernelized linear regression.

# Environment Setup
Installing the necessary packages
```
conda env create -f environment.yml
```
This will create a new environment called `icl`, which you can activate with:
```
conda activate icl
```

Setting up the jupyter notebook kernel for `eval.ipynb`
```
python -m ipykernel install --user --name=icl
```

If you want to train your own models on our scripts, please create a `models/` directory in the root directory of this repo. Therefore, the file structure should look like the following:
```
|--models/
|--src/
  |--conf/
  |--data/
  |--base_models.py
  |--...
|--environment.yml
|- ...
```

# Training
Our training setup for the models are stored in `src/conf/nanogpt_kernel_regression.yaml` and `src/conf/mamba_kernel_regression.yaml`. A toy configuration is stored in `src/conf/nanogpt.yaml` and `src/conf/mamba.yaml`.

The main parameters we have tweaked in these files include:
* `model.n_dims` - controls the dimension of the input `x` for each task
* `model.n_positions` - the max size of the context window for each task
* `training.task` - the in-context task to train the models on. For kernel linear regression, we have it set to `kernel_regression`
* `training.train_steps` - the number of batches to feed to the model during training
* `training.curriculum.points` - there are attributes for `start`, `end`, `inc`, and `interval`, which controls the start and end size of the in-context task prompt size and how much it will increase every `interval` training steps.
* `out_dir` - where you want the model to be saved. Default will save to `models/<task>/<model>`

To run the train script, run the following command in the `src/` directory:
```
python train.py --config <path_to_config.yaml>
```
To replicate our results, run these commands:
```
python train.py --config conf/nanogpt_kernel_regression.yaml

python train.py --config conf/mamba_kernel_regression.yaml
```
Note that you need to have access to CUDA cores to train.

# Evaluation
Our evaluations are ran in the `src/eval.ipynb` file. 

Please note that one of the cells will try to recompute the metrics with the assumption that there are models in a `models` directory (please refer to [Environment Setup](#environment-setup)). Either download our pretrained models (refer to [Our Results](#our-results)) or train your own. 

If you store your models somewhere else, change the `run_dir` variable at the top of the notebook.

# Our Results
To download our results, please download our models.zip file using this command:
```
wget https://github.com/joyjwlee/cs182_final_proj/releases/initial/download/initial/models.zip
unzip models.zip
```

# Credits
This repo was built off of the repo of Garg et al., linked [here](https://github.com/dtsip/in-context-learning).