"""
Build ensembles and calculate metrics:
python scripts/metrics_het_ensemble_parallel.py --dataset=cifar10
"""
import pandas as pd
from pathlib import Path
import numpy as np
from itertools import combinations
import random
from tqdm import tqdm
import multiprocessing
from functools import partial
import fire
import yaml
import os
from omegaconf import OmegaConf
from longtail_ensembles.utils_ens import read_model_size_file, get_ensembles_summary

BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
output_dir = BASE_DIR / "results"


def make_hom_groups(ind_models, ensemble_size=4):
    models = ind_models['models'].unique()
    ensemble_groups = set()

    for ii in range(len(models)):
        hom_models = ind_models[ind_models['models'] == models[ii]]
        model_comb = [frozenset(sorted(x)) for x in combinations(hom_models.index.values, ensemble_size)]
        [ensemble_groups.add(model_comb_) for model_comb_ in model_comb]
    return ensemble_groups


# read size of models trained on imagenet
def make_het_groups(ind_models, ensemble_size=4, binning=1, max_mpclass=11, max_enspclass=11):
  """
    Make heterogeneous ensembles by:
    - ensemble all models at random
    - ensemble all models in terms of performance
    - ensemble all models with bottom k models

    :param ind_models:
    :param ensemble_size:
    :param binning:
    :param max_mpclass:
    :param max_enspclass:
    :return:
    """
  # Options:

  # %% Filter out models in terms of performance
  err_values = 1 - ind_models['acc'].values
  # %%
  # To populate plot, we average model in different bins
  if binning == 1:
    # split models into 5 bins each of which has the same # models,
    err_bins = np.quantile(err_values, np.linspace(0, 1, 5))
  elif binning == 2:
    # split models according to their error level w same # models in each bin
    err_bins = np.histogram_bin_edges(err_values, 5)
  elif binning == 3:
    # split models where 10% are separated from the rest
    err_bins = np.quantile(err_values, 0.1)
  elif binning == 4:
    # split models into 10 bins -- equally spaced where max is the max error
    err_bins = np.linspace(0, err_values.max() + 0.01, 11)
  elif binning == 5:
    # split models into 10 bins -- equally spaced between 0-1
    err_bins = np.linspace(0, 1, 6)
  elif binning >= 10:
    # split models into 5 bins each of which has the same # models
    # but here 10+x where x is the # of models which come from worst performing models.
    # This to interpolate between the model lines
    err_bins = np.quantile(err_values, np.linspace(0, 1, 5))
  else:
    raise NotImplementedError('Binning {} not implemented'.format(binning))

  err_bins = np.unique(np.append(err_bins, np.asarray([0, 1])))

  model_cl_assignment = np.digitize(err_values, err_bins)
  model_classes, num_models_p_class = np.unique(model_cl_assignment, return_counts=True)
  print('Bins ', err_bins.round(2), num_models_p_class, flush=True)

  # %%
  all_ensemble_groups = set()
  for model_class_id in model_classes:
    # %% Find all models in group
    models_in_cls = np.argwhere(model_cl_assignment == model_class_id).flatten()
    if len(models_in_cls) <= ensemble_size:
      print('Skipping bin {}'.format(err_bins[model_class_id]), flush=True)

    # %% control number of models in each bin:
    if len(models_in_cls) > max_mpclass:
      models_in_cls = random.sample(list(models_in_cls), max_mpclass)
    # %%
    ensemble_groups = set()
    num_ensembles = min(max_enspclass, len(list(combinations(models_in_cls, ensemble_size))))
    while len(ensemble_groups) < num_ensembles:
      ensemble_groups.add(frozenset(sorted(random.sample(list(models_in_cls), ensemble_size))))
    try:
      err_bins[model_class_id]
    except:
      import pdb;
      pdb.set_trace()

    if binning >= 10:
      # replace to include worst case models
      worst_models = np.argsort(err_values)[::-1][:20]
      new_ensemble_groups = set()
      base_size = binning % 10
      worst_size = ensemble_size - base_size
      for _, ensemble_group in enumerate(ensemble_groups):
        l_models = random.sample(ensemble_group, base_size)
        m_models = random.sample(list(np.setdiff1d(worst_models, ensemble_group)), worst_size)
        new_group = l_models + m_models
        new_ensemble_groups.add(frozenset(new_group))
      # update so that we include at least a worst performance model
      ensemble_groups = new_ensemble_groups

    if len(ensemble_groups) == 0:
      print('Skipping bin {}'.format(err_bins[model_class_id]), flush=True)
      continue
    print('\nConstructed {} ensembles in bin {}'.format(len(ensemble_groups), err_bins[model_class_id]), flush=True)
    all_ensemble_groups = all_ensemble_groups.union(ensemble_groups)

  return all_ensemble_groups


def read_model_perf_file(dataset):
  if 'imagenet' in dataset:
    dataset = 'imagenet'
  elif dataset in ('cifar10', 'cinic10'):
    dataset = 'cifar10'
  model_size_file = BASE_DIR / "configs/datasets/{}/model_num_params.yaml".format(
    dataset)
  with open(model_size_file) as f:
    my_dict = yaml.safe_load(f)
  return my_dict


def get_player_score(perf):
  player_perf_bins = np.linspace(0, 1, 10)
  return np.digitize(perf, player_perf_bins)


# %%
def main(ensemble_size=4,
         max_mpclass=11,
         max_enspclass=11,
         serial_run=False,
         seed=0,
         dataset='cifar10',
         ensemble_type='het',
         out_name="model_performance/het_run_v20"):
  if ensemble_type == 'het':
    bins = [1, 2, 3]
  else:
    bins = [1]
  for binning in bins:
    run_model(binning=binning,
              ensemble_size=ensemble_size,
              max_mpclass=max_mpclass,
              max_enspclass=max_enspclass,
              serial_run=serial_run,
              seed=seed,
              ensemble_type=ensemble_type,
              dataset=dataset,
              out_name=out_name)


def run_model(ensemble_size=4,
              serial_run=False,
              dataset='cifar10',
              ensemble_type='het',
              out_name="model_performance/het_run_debug",
              binning=1,
              max_mpclass=11,
              max_enspclass=11,
              seed=None,
              ):
  # %%
  # get a random seed
  if seed is None:
    seed = np.random.randint(0, 100000)
  random.seed(seed)
  # %% results filename
  results_filename = output_dir / out_name / ('{}'.format(dataset) + ".csv")
  if os.path.exists(results_filename):
    df_results_base = pd.read_csv(results_filename, index_col=False)
    ensemble_groups_old = df_results_base['ensemble_group'].unique()
    ensemble_groups_old = set([frozenset([int(x) for x in y.split('--')]) for y in ensemble_groups_old])
  else:
    os.makedirs(results_filename.parent, exist_ok=True)

  if dataset in ('cifar10', 'cinic10', 'cifar10_1'):
    ind_dataset = 'cifar10'
  elif 'imagenet' in dataset:
    ind_dataset = 'imagenet'
  else:
    raise NotImplementedError('Dataset {} not implemented'.format(dataset))

  if dataset in ('cifar10', 'imagenet'):
    data_type = 'ind'
  else:
    data_type = 'ood'

  # %% read metrics_file
  metrics_file = BASE_DIR / "results/model_performance/{}.csv".format(
    dataset)
  metrics_ind_file = BASE_DIR / "results/model_performance/{}.csv".format(
    ind_dataset)

  ind_models = pd.read_csv(metrics_ind_file)
  file_models = pd.read_csv(metrics_file)

  assert ind_models.shape[0] == file_models.shape[0]

  # Build ensemble groups
  if ensemble_type == 'het':
    ensemble_groups = make_het_groups(ind_models,
                                      ensemble_size=ensemble_size,
                                      binning=binning,
                                      max_mpclass=max_mpclass,
                                      max_enspclass=max_enspclass)
  elif ensemble_type == 'hom':
    ensemble_groups = make_hom_groups(ind_models,
                                      ensemble_size=ensemble_size)
  else:
    raise NotImplementedError('Ensemble type {} not implemented'.format(ensemble_type))
  model_num_params = read_model_size_file(ind_dataset)['models']
  model_num_params = OmegaConf.to_container(model_num_params, resolve=True)

  # exclude models already run:
  if os.path.exists(results_filename):
    num_ensembles_new = len(ensemble_groups)
    ensemble_groups = ensemble_groups - ensemble_groups_old
    print('Excluding {} ensembles which already exist'.format(num_ensembles_new - len(ensemble_groups)), flush=True)

  if len(ensemble_groups) == 0:
    print('No new ensembles to run, exiting', flush=True)
    return
  print('\n Running {} ensembles'.format(len(ensemble_groups)), flush=True)

  # %%
  # run in parallel
  if serial_run:
    # """
    for egroup_idx, ensemble_group in enumerate(tqdm(ensemble_groups)):
      result = get_ensembles_summary(file_models,
                                     model_num_params,
                                     ensemble_group)
      result['ensemble_size'] = ensemble_size
      result['ensemble_type'] = ensemble_type
      result['seed'] = seed
      result['dataset'] = dataset
      result['data_type'] = data_type
      result['architecture'] = None
      result['binning'] = binning
      if os.path.exists(results_filename):
        df_results_base = pd.read_csv(results_filename, index_col=False)
        result = pd.concat([df_results_base, result], axis=0).drop_duplicates()
      result.reset_index(drop=True).to_csv(results_filename, index=False)

    # """
  else:
    get_ens = partial(get_ensembles_summary, file_models,
                      model_num_params)
    pool = multiprocessing.Pool(10)
    for result in tqdm(pool.imap_unordered(get_ens,
                                           ensemble_groups,
                                           chunksize=2),
                       total=len(ensemble_groups)):

      result['ensemble_size'] = ensemble_size
      result['ensemble_type'] = ensemble_type
      result['seed'] = seed
      result['dataset'] = dataset
      result['data_type'] = data_type
      result['architecture'] = None
      result['binning'] = binning
      if os.path.exists(results_filename):
        df_results_base = pd.read_csv(results_filename, index_col=False)
        result = pd.concat([df_results_base, result], axis=0).drop_duplicates()
      result.reset_index(drop=True).to_csv(results_filename, index=False)


  return


# %%
if __name__ == "__main__":
  fire.Fire(main)
