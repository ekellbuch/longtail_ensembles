"""
Test run with multiple gpus
"""
import os
from absl.testing import absltest
from absl.testing import parameterized
import sys
import yaml
import torch
from pathlib import Path
from omegaconf import OmegaConf
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

CIFAR10_CONFIG = BASE_DIR / "scripts/script_configs/run_cpu_cifar10.yaml"
CIFAR100_CONFIG = BASE_DIR / "scripts/script_configs/run_gpu_cifar100.yaml"

eval_configs = {
  'cifar10': CIFAR10_CONFIG,
  'cifar100': CIFAR100_CONFIG,
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


class ModuleTestRunCIFAR(parameterized.TestCase):

  @parameterized.named_parameters(
    # baseline
    ("base", "base", "cifar10", "CIFAR10",),
  )
  def test_imbalanced_losses(self, module, config_name, test_set):
    max_epochs = 10
    learning_rate = 0.01
    weight_decay = 0.01
    logger = None
    fast_dev_run = 1
    random_eval = 0
    seed = 1
    valid_size = 0.1
    return_model = 1

    gpus = torch.cuda.device_count()
    accelerator = None
    if gpus > 1:
      accelerator = "ddp"
    else:
      accelerator = "dp"

    args = create_cfg(config_name)
    args.module_cfg.module = "base"
    args.module_cfg.learning_rate = learning_rate
    args.module_cfg.weight_decay = weight_decay
    args.data_cfg.test_set = test_set
    args.data_cfg.valid_size = valid_size
    args.trainer_cfg.fast_dev_run = fast_dev_run
    args.trainer_cfg.logger = logger
    args.trainer_cfg.max_epochs = max_epochs
    args.trainer_cfg.gpus = gpus
    args.trainer_cfg.accelerator = accelerator
    args.eval_cfg.random_eval = random_eval
    args.eval_cfg.return_model = return_model
    args.seed = seed


    base_variables = train(args)

    trainer = base_variables[3]
    ind_data = base_variables[0]
    model = base_variables[2]
    ind_performance_base = trainer.test(model, ind_data.test_dataloader())

    args = create_cfg(config_name)
    args.module_cfg.module = module
    args.module_cfg.learning_rate = learning_rate
    args.module_cfg.weight_decay = weight_decay
    args.data_cfg.test_set = test_set
    args.data_cfg.valid_size = valid_size
    args.trainer_cfg.fast_dev_run = fast_dev_run
    args.trainer_cfg.logger = logger
    args.trainer_cfg.max_epochs = max_epochs
    args.trainer_cfg.gpus = gpus
    args.trainer_cfg.accelerator = accelerator
    args.eval_cfg.random_eval = random_eval
    args.eval_cfg.return_model = return_model
    args.seed = seed

    if module == "focal_loss":
      args.module_cfg.focal_size_average = True

    new_variables = train(args)

    trainer = new_variables[3]
    ind_data = new_variables[0]
    model = new_variables[2]
    ind_performance_new = trainer.test(model, ind_data.test_dataloader())

    self.assertAlmostEqual(ind_performance_base[0]['acc/test'],
                     ind_performance_new[0]['acc/test'], places=5)
    self.assertAlmostEqual(ind_performance_base[0]['loss/val'],
                     ind_performance_new[0]['loss/val'], places=5)


if __name__ == '__main__':
  absltest.main()