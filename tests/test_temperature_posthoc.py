"""
Test temperature scaling on trained models from the longtail_ensembles repo
python -m unittest -k tests.test_temperature.ModuleTest.test_no_acc_change_ts_vec
"""
from absl.testing import absltest
from absl.testing import parameterized
import os
import sys
import yaml
import torch
from pathlib import Path
from omegaconf import OmegaConf
from longtail_ensembles.temperature_scaling import ModuleWithTemperature, all_ts_modules
from pytorch_lightning import Trainer
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from torch.utils.data import Subset
import copy

NUM_CLASSES = {
  "CIFAR10" : 10,
  "CIFAR100" : 100,
  "IMBALANCECIFAR10" : 10,
  "IMBALANCECIFAR100" : 100,
}

BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.join(str(BASE_DIR),"scripts"))
from run import train

CIFAR10_CONFIG = BASE_DIR / "scripts/script_configs/eval_cifar10_fromrepo.yaml"
CIFAR100_CONFIG = BASE_DIR / "scripts/script_configs/eval_cifar100_fromrepo.yaml"
CIFAR10LT_TEMP = BASE_DIR / "scripts/script_configs/eval_cifar10lt_wtemp.yaml"

eval_configs = {
  'cifar10_noval': CIFAR10_CONFIG,
  'cifar100_noval': CIFAR100_CONFIG,
  'cifar10lt_wtmp': CIFAR10LT_TEMP,
}



def check_model_eq(model, model2):
  for param1, param2 in zip(model.parameters(), model2.parameters()):
    if param1.data.ne(param2.data).sum() > 0:
      return False
  return True

def create_cfg(eval_config_name) -> dict:
  """Load all toy data config file without hydra."""
  cfg = yaml.load(open(str(eval_configs[eval_config_name])), Loader=yaml.FullLoader)
  return OmegaConf.create(cfg)



class ModuleEnsembleTest(parameterized.TestCase):
  @parameterized.named_parameters(
    ("cifar10_base_ce", "cifar10_noval", 0.1, True, "ts_base", False),
  )
  def test_no_acc_change_ts(self, eval_config_name, valid_size, split_test, ts_module, use_train_loss):

    args = create_cfg(eval_config_name)
    args.data_cfg.num_workers = os.cpu_count()
    args.data_cfg.valid_size = valid_size
    args.trainer_cfg.fast_dev_run = 1
    args.trainer_cfg.logger = "tensorboard"
    args.eval_cfg.random_eval = 0
    args.eval_cfg.return_model = 1

    # apply temperature scaling
    args.module_cfg.temperature_scaling = 1
    args.ts_cfg.use_train_loss = use_train_loss
    args.ts_cfg.ts_module = ts_module
    #args.trainer_cfg.accelerator = "cpu"

    ind_data, ood_data, model, trainer = train(args)
    if split_test:
      # Split test set if val_set=test_set during training
      num_test = len(ind_data.test_dataset)
      indices = torch.randperm(num_test)
      split = int(np.floor(num_test*valid_size))
      new_val_indices = indices[:split]
      new_test_indices = indices[split:]

      print('New train and val indices : {}, {}'.format(len(new_val_indices), len(new_test_indices)))
      ind_data.val_dataset = Subset(ind_data.test_dataset, new_val_indices)
      ind_data.test_dataset = Subset(ind_data.test_dataset, new_test_indices)

    metrics_wo_ts = trainer.test(model, ind_data.test_dataloader())

    # evaluate performance with temperature scaling
    if bool(args.module_cfg.temperature_scaling):
        ts_model = all_ts_modules[args.ts_cfg.ts_module](model,
                                          init_temp=args.ts_cfg.init_temp,
                                          use_train_loss=args.ts_cfg.use_train_loss,
                                          opt_params=args.ts_cfg.get('opt_params', None),
                                          )
        ts_trainerargs = copy.deepcopy(OmegaConf.to_container(args.trainer_cfg))
        ts_trainerargs['logger'] = None
        ts_trainerargs['max_epochs'] = args.ts_cfg.max_epochs
        # estimate the temperature
        ts_trainer = Trainer(**ts_trainerargs)

        ts_trainer.fit(model=ts_model, train_dataloaders=ind_data.val_dataloader())
        ts_model.temperature.requires_grad = False
        # set temperature
        print('\ntemperature scaling factor: T={}'.format(ts_model.temperature.numpy()), flush=True)
        model.model = ModuleWithTemperature(model.model, ts_model.temperature)
        # evaluate performance with temperature scaling
        metrics_w_ts = trainer.test(model, ind_data.test_dataloader())

        self.assertEqual(metrics_wo_ts[0]['acc/test'],
                         metrics_w_ts[0]['acc/test'])
        self.assertGreaterEqual(metrics_wo_ts[0]['acc/test'],
                                metrics_w_ts[0]['acc/test'])
        self.assertGreaterEqual(metrics_wo_ts[0]['ece/test'],
                                metrics_w_ts[0]['ece/test'])


class ModuleTest(parameterized.TestCase):
  @parameterized.named_parameters(
    ("cifar10_base_ce", "cifar10_noval", 0.1, True, "ts_base", False),
    ("cifar10_base", "cifar10_noval", 0.1, True, "ts_base", True),
    ("cifar10_vector_ce", "cifar10_noval", 0.1, True, "ts_vector", False), #False
    ("cifar10_vector", "cifar10_noval", 0.1, True, "ts_vector", True), # False
    #("eval_cifar100_repo", "cifar100_noval", 0.1, True),
    #("eval_cifar10lt_tmp", "cifar10lt_wtmp", 0.1, False),
  )
  def test_no_acc_change_ts(self, eval_config_name, valid_size, split_test, ts_module, use_train_loss):

    args = create_cfg(eval_config_name)
    args.data_cfg.num_workers = os.cpu_count()
    args.data_cfg.valid_size = valid_size
    args.trainer_cfg.fast_dev_run = 1
    args.trainer_cfg.logger = "tensorboard"
    args.eval_cfg.random_eval = 0
    args.eval_cfg.return_model = 1
    # apply temperature scaling
    args.module_cfg.temperature_scaling = 1
    args.ts_cfg.use_train_loss = use_train_loss
    args.ts_cfg.ts_module = ts_module
    #args.trainer_cfg.accelerator = "cpu"

    ind_data, ood_data, model, trainer = train(args)
    if split_test:
      # Split test set if val_set=test_set during training
      num_test = len(ind_data.test_dataset)
      indices = torch.randperm(num_test)
      split = int(np.floor(num_test*valid_size))
      new_val_indices = indices[:split]
      new_test_indices = indices[split:]

      print('New train and val indices : {}, {}'.format(len(new_val_indices), len(new_test_indices)))
      ind_data.val_dataset = Subset(ind_data.test_dataset, new_val_indices)
      ind_data.test_dataset = Subset(ind_data.test_dataset, new_test_indices)

    metrics_wo_ts = trainer.test(model, ind_data.test_dataloader())

    # evaluate performance with temperature scaling
    if bool(args.module_cfg.temperature_scaling):
        ts_model = all_ts_modules[args.ts_cfg.ts_module](model,
                                          init_temp=args.ts_cfg.init_temp,
                                          use_train_loss=args.ts_cfg.use_train_loss,
                                          opt_params=args.ts_cfg.get('opt_params', None),
                                          )
        ts_trainerargs = copy.deepcopy(OmegaConf.to_container(args.trainer_cfg))
        ts_trainerargs['logger'] = None
        ts_trainerargs['max_epochs'] = args.ts_cfg.max_epochs
        # estimate the temperature
        ts_trainer = Trainer(**ts_trainerargs)

        ts_trainer.fit(model=ts_model, train_dataloaders=ind_data.val_dataloader())
        ts_model.temperature.requires_grad = False
        # set temperature
        print('\ntemperature scaling factor: T={}'.format(ts_model.temperature.numpy()), flush=True)
        model.model = ModuleWithTemperature(model.model, ts_model.temperature)
        # evaluate performance with temperature scaling
        metrics_w_ts = trainer.test(model, ind_data.test_dataloader())

        self.assertEqual(metrics_wo_ts[0]['acc/test'],
                         metrics_w_ts[0]['acc/test'])
        self.assertGreaterEqual(metrics_wo_ts[0]['acc/test'],
                                metrics_w_ts[0]['acc/test'])
        self.assertGreaterEqual(metrics_wo_ts[0]['ece/test'],
                                metrics_w_ts[0]['ece/test'])


if __name__ == '__main__':
  absltest.main()