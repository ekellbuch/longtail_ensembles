"""
Losses should be the same if the classes are balanced
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


class ModuleTestLossRunCIFAR10(parameterized.TestCase):

  @parameterized.named_parameters(
    # baseline
    #("base", "base", "cifar10", "CIFAR10",),
    #("base_imbcifar100", "base", "cifar100", "IMBALANCECIFAR100",),
    #("base_imbcifar10", "base", "cifar100", "IMBALANCECIFAR10",),
    # losses for imbalanced classes:
    #("weighted_ce", "weighted_ce", "cifar10", "CIFAR10",),
    #("base_bloss", "base_bloss", "cifar10", "CIFAR10",),
    ("focal_loss", "focal_loss", "cifar10", "CIFAR10",),
    #("weighted_softmax", "weighted_softmax", "cifar10", "CIFAR10",),
    #("base_cifar100", "base", "cifar100", "CIFAR100",),
    #("weighted_ce_cifar100", "weighted_ce", "cifar100", "CIFAR100",),
    #("base_bloss_cifar100", "base_bloss", "cifar100", "CIFAR100",),
    #("weighted_softmax_cifar100", "weighted_softmax", "cifar100", "CIFAR100",),
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
    return_model =  1

    args = create_cfg(config_name)
    args.module_cfg.module = "base"
    args.module_cfg.learning_rate = learning_rate
    args.module_cfg.weight_decay = weight_decay
    args.data_cfg.test_set = test_set
    args.data_cfg.valid_size = valid_size
    args.trainer_cfg.fast_dev_run = fast_dev_run
    args.trainer_cfg.logger = logger
    args.trainer_cfg.max_epochs = max_epochs
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


if __name__ == '__main__':
  absltest.main()