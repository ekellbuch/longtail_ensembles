#!/usr/bin/env python
"""
python make logit yaml file

python scripts/make_logit_yaml.py --sweep=1q8q8q8q
"""
import fire
import wandb
import os
import sys
from pathlib import Path
from omegaconf import ListConfig, OmegaConf
import wandb
module_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(module_dir, "scripts/run"))
import run as run_script

#%%
def main_v0(sweep_name="6o3j1rcj"):
  print('not here')
  #%%
  api = wandb.Api()
  sweep = api.sweep("ekellbuch/uncategorized/sweeps/{}".format(sweep_name))
  sweep_runs = sweep.runs

  exp_keys = ['module', 'out_checkpoint_dir', 'seed', 'test_set', 'ood_dataset']

  #%%
  for run in sweep_runs:
    print(run.summary)
    main_path = run.config['out_logits_dir']
    labelpaths = os.path.join(main_path, 'ind_labels.npy')
    filepaths = os.path.join(main_path, 'model.pt')
    print("- {}".format(labelpaths))
    print("- {}".format(filepaths))

  #%%

  return


def main(sweep_name="6o3j1rcj"):
  "store all logits given sweep"
  #%%
  api = wandb.Api()
  sweep = api.sweep("ekellbuch/uncategorized/sweeps/{}".format(sweep_name))
  sweep_runs = sweep.runs

  exp_keys = ['module', 'out_checkpoint_dir', 'seed', 'test_set', 'ood_dataset']

  cwd = os.getcwd()
  #%%
  for run in sweep_runs:
    #print(run.summary)
    try:
      main_path = run.config['out_logits_dir']
    except:
      pass
    try:
      ckpt_path = run.config['out_checkpoint_dir']
      project_name = '{}_{}_{}'.format(run.config['project_name'], sweep_name, 'logits')
      data = {}
      data['checkpoint'] = ckpt_path
      data['test_phase'] = 1
      data['fast_dev_run'] = 0
      data['project_name'] = project_name

      #%%
      logits_fdir, logits_bdir = ckpt_path.split('models')
      os.chdir(logits_fdir)
      new_config = OmegaConf.create(run.config)
      # get checkpoint_dir
      assert wandb.run is None
      wandb.init(project=project_name)
      new_config.update(data)
      run_script.train(new_config)
      print('Compare with:')
      print(run.summary['acc/val'])

      wandb.run.finish()
      assert wandb.run is None
      os.chdir(cwd)
    except:
      pass
  #%%
  return

if __name__ == "__main__":
    fire.Fire(main)

