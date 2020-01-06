# Semi-Supervised Learning with Scarce Annotations
Code to reproduce some of the main results in:
> Sylvestre-Alvise Rebuffi, Sebastien Ehrhardt, Kai Han, Andrea Vedaldi, Andrew Zisserman, "Semi-Supervised Learning with Scarce Annotations", [arXiv](https://arxiv.org/pdf/1905.08845.pdf)

## Requirements
Install requirements: the environement used to run this code is provided in `environment.yml`. It can be installed using conda with the following command (environment name will be `salsa`):

`conda env create -f environment.yml`

## Train a RotNet
`python rotNet.py --dataset mydataset --network mynetwork --save_dir myrotnetdir`

Choose any network among: {ResNet-18, RevNet-18,TempEns} and dataset in {cifar10, cifar100, svhn}

## Alternative training with semi-supervision
`python alternative_training.py --dataset mydataset --save_dir mydir --rotnet_dir myrotnetdir --nb_labels_per_class 10`

Default parameters are for CIFAR10. For CIFAR100 use inner milestones [14,20], for SVHN use learning rate of 0.1 and outer milestones [120,150].

## Training scripts
A sample training script  to run the same experiments 10 time with different dataset splits is available in `scripts/`. You will have to specify (in the following order) dataset, number of labels per class, save_dir, and rotnet_dir.

For instance: `sh ./scripts/train_semi.sh cifar10 10 mydir myrotnetdir`

## Train with full supervision
`python supervised_training.py --dataset mydataset --network mynetwork --save_dir mydir --rotnet_dir myrotnetdir`

## Available Datasets
This code supports CIFAR10, CIFAR100 and SVHN datasets.

## Two moons figure
We also provide the script to generate the two moons figure of the paper (Fig 1.). To generate the pictures run `python two_moons/pi_model.py`, figures will be available in the folder `render/`.

## Cite this work
If you use this code for your project please consider citing us:
```
@article{rebuffi2019semi,
  title={Semi-Supervised Learning with Scarce Annotations},
  author={Rebuffi, Sylvestre-Alvise and Ehrhardt, Sebastien and Han, Kai and Vedaldi, Andrea and Zisserman, Andrew},
  journal={Technical report},
  year={2019}
}
```
