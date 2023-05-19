from longtail_ensembles.predictions import EnsembleModel, EnsembleModelLogit, MultipleModel
from itertools import combinations
import os
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import yaml
import numpy as np

ensemble_classes = {
  "no_avg": MultipleModel,  # multiple models where metrics are avg of single model metrics
  "avg_probs": EnsembleModel,  # ensemble formed by averaging probabilities
  "avg_logits": EnsembleModelLogit,  # ensemble formed by averaging logits
}

BASE_DIR = Path(os.path.abspath(os.path.dirname(__file__))).parents[1]


def get_all_ensemble_pairs(model, ensemble_size=4):
  ensemble_groups = list(combinations(range(len(model["filepaths"])), ensemble_size))
  all_new_model_pairs = []
  for ensemble_group in ensemble_groups:
    new_models = {}
    new_models['filepaths'] = np.asarray(model["filepaths"])[[ensemble_group]].tolist()[0]
    new_models['labelpaths'] = np.asarray(model["labelpaths"])[[ensemble_group]].tolist()[0]
    new_models['npz_flag'] = model.get('npz_flag', None)
    all_new_model_pairs.append(new_models)
  return all_new_model_pairs


def build_ensemble(model,
                   ensemble_method="avg_logits",
                   ensemble_size=4,
                   file_ext=None,
                   label_ext=None,
                   ):
  ensemble_groups = list(combinations(range(len(model["filepaths"])), ensemble_size))
  # TODO: makes only 1 combination of models for now
  ensemble_group = ensemble_groups[0]
  models_ = [model["filepaths"][ens_mem] for ens_mem in ensemble_group]
  labels_ = [model["labelpaths"][ens_mem] for ens_mem in ensemble_group]

  ensemble_class = ensemble_classes[ensemble_method]
  ens1 = ensemble_class(ensemble_method, "ind")
  npz_flag = model.get('npz_flag', None)
  for i, (m, l) in enumerate(zip(models_, labels_)):
    m = os.path.join(m, file_ext) if file_ext is not None else m
    l = os.path.join(l, label_ext) if label_ext is not None else l
    ens1.register(filename=m,
                  modelname=i,
                  inputtype=None,
                  labelpath=l,
                  npz_flag=npz_flag)

  return ens1


def get_model_metrics(model,
                      modelname=None):
  """
  Takes as argument model class
  keys giving model names, values are dictionaries with paths to individual entries.
  :param models: names of individual models.
  """

  all_metrics = []

  acc, brier, ece, f1, nll, calibauc, calipr, ens_var, cls_var \
    = model.get_accuracy(),  model.get_brier(), model.get_ece(), model.get_f1score(), model.get_nll(), \
    model.get_calibration_roc_auc(), model.get_calibration_pr_auc(),\
    model.get_variance(), model.get_class_variance()
  print("{}: Acc: {:.3f}, Brier: {:.3f}, ECE: {:.3f}, F1: {:.3f}, NLL: {:.3f} CalROCauc: {:.3f} CalPRauc: {:.3f} EnsVar: {:.3f} ClsVar: {:.3f}".
        format(modelname, acc, brier, ece, f1, nll, calibauc, calipr, ens_var, cls_var), flush=True)
  all_metrics.append([modelname, acc, brier, ece, f1, nll, calibauc, calipr, ens_var, cls_var])

  df = pd.DataFrame(all_metrics,
                    columns=[
                      "models", "acc", "brier", "ece",
                      "f1", "nll", "CalROCauc", "CalPRauc", "EnsVar", "ClsVar"
                    ])

  return df


def get_model_metrics_pclass(model,
                      modelname=None,
                      num_classes=None):
  """
  Takes as argument model class
  keys giving model names, values are dictionaries with paths to individual entries.
  :param models: names of individual models.
  """

  acc_pclass = model.get_accuracy_per_class(num_classes=num_classes)
  f1_score_pclass = model.get_f1score_per_class(num_classes=num_classes)
  # var across models, sum over classes: spread in predictions
  var_score_pclass = model.get_variance_per_class()
  cv_score_pclass = model.get_cv_per_class()
  # given final model prediction var across classes, average over samples
  cvar_score_pclass = model.get_class_variance_per_class()
  disagree_score_pclass = model.get_diversity_score_per_class(metric='avg_disagreement')

  df1 = pd.DataFrame.from_dict(acc_pclass, orient='index', columns=['acc'])
  df2 = pd.DataFrame.from_dict(f1_score_pclass, orient='index', columns=['f1'])
  df3 = pd.DataFrame.from_dict(var_score_pclass, orient='index', columns=['var'])
  #df4 = pd.DataFrame.from_dict(cv_score_pclass, orient='index', columns=['cv'])
  df5 = pd.DataFrame.from_dict(cvar_score_pclass, orient='index', columns=['class_var'])
  df6 = pd.DataFrame.from_dict(disagree_score_pclass, orient='index', columns=['avg_disagreement'])
  df = pd.concat([df1, df2, df3, df5, df6], axis=1)
  df['Class ID'] = df.index
  df.reset_index(drop=True, inplace=True)
  df['models'] = modelname

  return df


def read_model_size_file(dataset):
  if 'imagenet' in dataset:
    dataset = 'imagenet'
  elif dataset in ('cifar10', 'cinic10'):
    dataset = 'cifar10'
  model_size_file = BASE_DIR / "configs/datasets/{}/model_num_params.yaml".format(
    dataset)
  cfg = yaml.load(open(str(model_size_file)), Loader=yaml.FullLoader)
  return OmegaConf.create(cfg)


def get_ensembles_summary(ind_models, model_num_params, ensemble_group):
  ensemble_types = ['no_avg', 'avg_logits', 'avg_probs']
  ensemble_group = np.sort(np.asarray(list(ensemble_group)))
  model_names = ind_models.loc[ensemble_group].models.values

  # get ens param count
  model_param_count = 0
  for model_name_ in model_names:
    model_param_count += model_num_params[model_name_]['num_params']

  model_names = ind_models.loc[ensemble_group].models.values
  name = "--".join([m_ for m_ in model_names]) + "--"
  model_ids = "--".join([str(m_) for m_ in ensemble_group])
  name += model_ids

  all_metrics = []
  models = ind_models.loc[ensemble_group]

  for ensemble_type_ in ensemble_types:
    # build ensemble
    new_models = {}
    new_models['filepaths'] = models.filepaths.values
    new_models['labelpaths'] = models.labelpaths.values
    new_models['npz_flag'] = models.npz_flag.values[0]
    ens = build_ensemble(new_models,
                         ensemble_method=ensemble_type_)
    try:
      metrics = get_model_metrics(ens, modelname=name, )
    except:
      raise Warning('Failed to ensemble {}'.format(name))
      continue
      #
    metrics['model_scores'] = None
    metrics['ensemble_avg_type'] = ensemble_type_
    metrics['num_params'] = model_param_count
    metrics['ensemble_group'] = model_ids
    all_metrics.append(metrics)

    del ens
  all_metrics = pd.concat(all_metrics)
  return all_metrics

def get_ensembles_summary_pclass(ind_models, model_num_params, num_classes, ensemble_group):
  ensemble_types = ['no_avg', 'avg_logits', 'avg_probs']
  ensemble_group = np.sort(np.asarray(list(ensemble_group)))
  model_names = ind_models.loc[ensemble_group].models.values

  # get ens param count
  model_param_count = 0
  for model_name_ in model_names:
    model_param_count += model_num_params[model_name_]['num_params']

  model_names = ind_models.loc[ensemble_group].models.values
  name = "--".join([m_ for m_ in model_names]) + "--"
  model_ids = "--".join([str(m_) for m_ in ensemble_group])
  name += model_ids

  all_metrics = []
  models = ind_models.loc[ensemble_group]

  for ensemble_type_ in ensemble_types:
    # build ensemble
    new_models = {}
    new_models['filepaths'] = models.filepaths.values
    new_models['labelpaths'] = models.labelpaths.values
    new_models['npz_flag'] = models.npz_flag.values[0]
    ens = build_ensemble(new_models,
                         ensemble_method=ensemble_type_)
    try:
      metrics = get_model_metrics_pclass(ens, modelname=name, num_classes=num_classes)
    except:
      raise Warning('Failed to ensemble {}'.format(name))
      continue
      #
    metrics['model_scores'] = None
    metrics['ensemble_avg_type'] = ensemble_type_
    metrics['num_params'] = model_param_count
    metrics['ensemble_group'] = model_ids
    all_metrics.append(metrics)


  all_metrics = pd.concat(all_metrics)
  return all_metrics