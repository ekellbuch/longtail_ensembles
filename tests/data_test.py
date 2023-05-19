
from absl.testing import absltest
from ml_collections import config_dict
from pathlib import Path
from absl.testing import parameterized
import os
import sys
import yaml
import pytest
import torch
from pathlib import Path
import copy
import pytorch_lightning as pl
from longtail_ensembles.data import all_datasets
from longtail_ensembles.module import all_modules
from omegaconf import OmegaConf, open_dict

from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = {
  "CIFAR10" : 10,
  "CIFAR100" : 100,
  "IMBALANCECIFAR10" : 10,
  "IMBALANCECIFAR100" : 100,
}

BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.join(str(BASE_DIR),"scripts"))
from run import train


if torch.cuda.is_available():
  TOY_CONFIG = BASE_DIR / "scripts/script_configs/run_gpu_cifar10.yaml"
else:
  TOY_CONFIG = BASE_DIR / "scripts/script_configs/run_cpu_cifar10.yaml"


def create_cfg() -> dict:
  """Load all toy data config file without hydra."""
  cfg = yaml.load(open(str(TOY_CONFIG)), Loader=yaml.FullLoader)
  return OmegaConf.create(cfg)


def check_model_eq(model, model2):
  for param1, param2 in zip(model.parameters(), model2.parameters()):
    if param1.data.ne(param2.data).sum() > 0:
      return False


def compare_loader(loader1, loader2, mode='test', shuffle=False, aug=True):
  loader_modes = {
    'train': lambda x : x.train_dataloader(shuffle=shuffle, aug=aug),
    'val': lambda x: x.val_dataloader(),
    'test': lambda x: x.test_dataloader(),
  }
  loader_mode = loader_modes[mode]
  with torch.no_grad():
    for idx, (batch1, batch2) in tqdm(enumerate(zip(loader_mode(loader1), loader_mode(loader2)))):
      if not torch.all(torch.eq(batch1[0], batch2[0])):
        return False
  return True


class ModuleTest(parameterized.TestCase):

  @parameterized.named_parameters(
    ("seed_gl0loc0_cifar10", "CIFAR10",0, None, None, "train", 1),
    ("seed_gl0loc0_imbcifar10", "IMBALANCECIFAR10",0, None, None, "train", 0),
    ("seed_gl1loc0_cifar10", "CIFAR10", 0, 1, None, "train", True),
    ("seed_gl1loc0_cifar10split", "CIFAR10", 0.1, 1, None, "train", True),
    ("seed_gl1loc0_imbcifar10", "IMBALANCECIFAR10", 0, 1, None, "train", True),
    ("seed_gl1loc1_imbcifar100", "IMBALANCECIFAR100", 0, 1, 1, "train", True),
    ("seed_gl0loc0_cifar10val", "CIFAR10", 0, None, None, "val", 1),
    ("seed_gl1loc0_cifar10val", "CIFAR10", 0.1, 1, None, "val", True),
    ("seed_gl1loc1_cifar10splitval", "CIFAR10", 0.1, 1, 1, "val", True),
  )
  def test_run_data_train(self, dataset, valid_size, global_seed, data_seed, mode, data_equal):

    args = create_cfg()
    args.module_cfg.module = "base"
    args.seed = global_seed
    args.data_cfg.valid_size = valid_size
    args.data_cfg.seed = data_seed
    args.trainer_cfg.fast_dev_run = 1
    args.trainer_cfg.logger = "tensorboard"
    args.eval_cfg.return_model = 1
    args.data_cfg.test_set = dataset
    args.data_cfg.num_classes = NUM_CLASSES[dataset]

    for run_idx in range(2):
      ind_data, _, _, _ = train(args)
      if run_idx == 0:
        new_data = ind_data
      else:
        all_pass = compare_loader(ind_data, new_data, mode=mode)

        self.assertEqual(all_pass, data_equal,
                         f'with seed global={global_seed} local={data_seed},'
                         f'data equality = {all_pass} in {mode} instead of {data_equal}')

  @parameterized.named_parameters(
     ("data_no_seed", "base", 0, 2, None, "test",1),
     ("data_seed", "base", 0, 2, 1, "test", 1),
     ("data_no_seed_split", "base", 0.1, 2, None, "test", 1),
     ("data_seed_split", "base", 0.1, 2, 1, "test", 1),
     ("data_seed_splitL", "base", 0.5, 2, 2, "train", 1)
  )
  def test_data_loader(self, module, valid_size, n_runs, seed, mode, data_equal):
    # Check train set is the same given different runs
    print(f'run {module}{valid_size}{n_runs}{seed}{data_equal}')
    args = create_cfg()
    args.module_cfg.module = module
    args.seed = seed
    args.data_cfg.valid_size = valid_size

    for run_idx in range(n_runs):
      ind_data = all_datasets[args.data_cfg.test_set](args.data_cfg)
      ind_data.setup()

      if run_idx == 0:
        new_data = ind_data
      else:
        all_pass = compare_loader(ind_data, new_data, mode=mode)

        self.assertEqual(all_pass, data_equal,
                         f'with seed {seed}, data equality = {all_pass} in {mode} instead of {data_equal}')
  

  @parameterized.named_parameters(
     # all train data
     ("seed_global_0_local_0", "base", 0, None, None, "train", 0),
     ("seed_global_0_local_0_subset", "base", 0.1, None, None, "train", 0),
     ("seed_global_0_local_1_subset", "base", 0.1, None, 2, "train", 1),  # subset is the same with data_seed
     ("seed_global_1_local_0_subset", "base", 0.1, 2, None, "train", 1),  # subset is the same with global_seed
  )
  def test_train_data_loader(self, module, valid_size, global_seed, data_seed, mode, data_equal):

    args = create_cfg()
    args.module_cfg.module = module
    args.seed = global_seed
    args.data_cfg.valid_size = valid_size
    args.data_cfg.seed = data_seed

    for run_idx in range(2):
      if global_seed is not None:
        pl.seed_everything(global_seed)

      ind_data = all_datasets[args.data_cfg.test_set](args.data_cfg)
      ind_data.setup()

      if run_idx == 0:
        new_data = ind_data
      else:
        all_pass = compare_loader(ind_data, new_data, mode=mode)

        self.assertEqual(all_pass, data_equal,
                         f'with seed global={global_seed} local={data_seed},'
                         f'data equality = {all_pass} in {mode} instead of {data_equal}')


  @parameterized.named_parameters(
     ("ensemble_no_seed", "ensemble", 3, None, 0),
     ("ensemble_seed", "ensemble", 3, 0, 0),
  )
  def test_ensemble_match_parameter(self, module, nb_models, seed, model_equal):
    # all members of the ensemble should be the same parameters
    args = create_cfg()
    args.seed = seed
    args.module_cfg.module = module
    args.module_cfg.nb_models = nb_models

    module_args = args.module_cfg
    OmegaConf.set_struct(module_args, True)
    with open_dict(module_args):
        module_args.num_classes = args.data_cfg.num_classes

    module_args = {"hparams": module_args}
    model = all_modules[args.module_cfg.module](**module_args)

    # compare model weights:
    num_eq_model = 0
    for idx, model in enumerate(model.models):
      if idx == 0:
        model2 = model
      else:
        model_eq = check_model_eq(model, model2)
        num_eq_model += 1 if model_eq == 1 else 0

    all_eq = num_eq_model == nb_models - 1

    self.assertEqual(all_eq, model_equal,
                     'Expected {} equal models , got {}'.format(model_equal, num_eq_model))


if __name__ == '__main__':
  absltest.main()