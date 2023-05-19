from absl.testing import absltest
from absl.testing import parameterized
from longtail_ensembles.metrics import AccuracyData
import numpy as np
from longtail_ensembles.predictions import EnsembleModel, EnsembleModelLogit, MultipleModel
import os
from pathlib import Path
import yaml
import sys
BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from longtail_ensembles.utils_ens import build_ensemble, get_model_metrics, read_model_size_file
from omegaconf import OmegaConf
import pandas as pd
ensemble_classes = {
  "no_avg": MultipleModel,  # multiple models where metrics are avg of single model metrics
  "avg_probs": EnsembleModel,  # ensemble formed by averaging probabilities
  "avg_logits": EnsembleModelLogit,  # ensemble formed by averaging logits
}

def read_dataset_yaml(ind_dataset):
  # % read metrics_file
  if 'imagenet' in ind_dataset:
    data_dir = BASE_DIR / "configs/datasets/{}".format('imagenet')
    ood_dataset = 'imagenetv2mf'
  elif 'cifar10' in ind_dataset:
    data_dir = BASE_DIR / "configs/datasets/{}".format('cifar10')
    ood_dataset = 'cinic10'

  ind_dataset_file = str(data_dir / "{}.yaml".format(ind_dataset))
  ood_dataset_file = str(data_dir / "{}.yaml".format(ood_dataset))

  with open(ind_dataset_file) as f:
    my_ind_dict = yaml.safe_load(f)

  with open(ood_dataset_file) as f:
    my_ood_dict = yaml.safe_load(f)
  my_ind_dict =OmegaConf.create(my_ind_dict)
  my_ood_dict =OmegaConf.create(my_ood_dict)

  model_num_params = read_model_size_file(ind_dataset)['models']
  model_num_params = OmegaConf.to_container(model_num_params, resolve=True)

  return my_ind_dict, my_ood_dict, model_num_params


class ModuleTestBuildEnsemble(parameterized.TestCase):
  # worse are lenet
  @parameterized.named_parameters(
    ('acc_hom_model', "cifar10"),
   )
  def test_metrics(self, dataset):

    ind_models, ood_models, model_num_params = read_dataset_yaml(dataset)
    all_models = ind_models.models.keys()
    all_metrics = []
    for model_name in all_models:
      print(model_name)
      local_models = {
        'ind': ind_models.models[model_name],
        'ood': ood_models.models[model_name]
      }
      ensemble_size = 3
      model_param_count = model_num_params[model_name]['num_params']*ensemble_size
      ensemble_types = ['no_avg', 'avg_logits', 'avg_probs']
      # For each ensemble type, register models
      for ensemble_type_ in ensemble_types:
        for data_type in local_models.keys():
          models = local_models[data_type]
          if len(models['filepaths']) < ensemble_size:
            continue

          ens = build_ensemble(model=models,
                               ensemble_method=ensemble_type_,
                               ensemble_size=ensemble_size,
                               )
          metrics = get_model_metrics(ens,
                                      modelname=model_name)
          #metrics['model_scores'] = None
          metrics['ensemble_avg_type'] = ensemble_type_
          metrics['data_type'] = data_type
          #metrics['num_params'] = model_param_count
          metrics['0-1_error'] = 1 - metrics['acc']
          all_metrics.append(metrics)

    all_metrics = pd.concat(all_metrics)

    print("\n\n")
    print(all_metrics)
    #tmp_metrics = all_metrics.copy()
    #tmp_metrics = tmp_metrics[(tmp_metrics["data_type"] == "ind") & (tmp_metrics["0-1_error"] >= 0.15)]
    #tmp_metrics = tmp_metrics[(tmp_metrics["data_type"] == "ind") & (tmp_metrics["nll"] >= 0.6)]
    #print(tmp_metrics)
    # what is a homogeneous ensemble of lenet models?
    import pdb; pdb.set_trace()
    return all_metrics


if __name__ == '__main__':
  absltest.main()
