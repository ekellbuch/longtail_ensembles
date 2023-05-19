# The Effects of Ensembling on Long-Tailed Data

Code for the paper "The Effects of Ensembling on Long-Tailed Data"

```
@inproceedings{
buchanan2023the,
title={The Effects of Ensembling on Long-Tailed Data},
author={E. Kelly Buchanan and Geoff Pleiss and Yixin Wang and John Patrick Cunningham},
booktitle={NeurIPS 2023 Workshop Heavy Tails in Machine Learning},
year={2023}
}
```
Installation instructions in docs/README.md: [docs/README.md](docs/README.md)

# Experiments:
1. Train resnet32 model on CIFAR10 dataset
```
python scripts/run.py --config-name="run_gpu_cifar10"
```
2. Train resnet32 and resnet110 model on CIFAR10LT dataset across multiple losses
```
wandb sweep experiments/compare_loss/train_gpu_loss_cifar10.yaml
```
3. Train resnet32 model on CIFAR10LT dataset across multiple losses
```
wandb sweep experiments/compare_loss/train_gpu_loss_cifar10_largeM.yaml
```

# Paper Experiments

|                                                                                                 Wandb Experiment                                                                                                  |                                            parameters                                            |                                                                                                                                     comments                                                                                                                                      |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [nggmmw4m](https://wandb.ai/ekellbuch/uncategorized/sweeps/nggmmw4m) , [0itowy8a](https://wandb.ai/ekellbuch/uncategorized/sweeps/0itowy8a), [d4s9wp4v](https://wandb.ai/ekellbuch/uncategorized/sweeps/d4s9wp4v) | train resnet32 and resnet110 models on CIFAR10-LT using multiple losses and for different seeds. |     models trained using balanced softmax loss have best performance                                                                                                  
|                       [9hwaytks](https://wandb.ai/ekellbuch/longtail_ensembles-scripts/sweeps/9hwaytks), [gv4bucon](https://wandb.ai/ekellbuch/longtail_ensembles-scripts/sweeps/gv4bucon)                        | train resnet32_cfa and resnet_110 on CIFAR100-LT using multiple losses and for difference seeds. |   models trained using balanced softmax loss have best performance        


# Reproduce paper tables and figures:
- [x] Fig: Ensemble size vs ensemble type across multiple losses
```
python scripts/vis_scripts/plot_results_metric_M.py --config-path="../../configs/comparison_baseline_cifar10lt" --config-name="compare_M"
```
- [x] Table: Ensemble performance of models trained on  CIFAR10-LT and CIFAR100-LT:
```
python scripts/compare_all_results.py --config-path="../configs/comparison_baseline_cifar10lt" --config-name="default"
python scripts/compare_all_results.py --config-path="../configs/comparison_baseline_cifar100lt" --config-name="default"
```
- [x] Fig: Class ID vs avg. Disagreement:
```
python scripts/vis_scripts/plot_results_pclass.py 
```
- [x] Fig: Class ID vs diversity/dependency:
```
python scripts/vis_scripts/plot_results_dkl_diff.py 
```
- [x] Fig: performance of logit and probability ensembles on balanced datasets. 
```
python scripts/vis_scripts/plot_single_metric_xy.py --datasets=base --metric=error
```

# Other implementations

-  Balanced Meta Softmax: [github.com/jiawei-ren/BalancedMetaSoftmax-Classification](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification)
