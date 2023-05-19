"""
Given a config as compare_results_cifar100_trial1.yaml
calculate the metrics for different ensemble types

python scripts/compare_results.py

python scripts/compare_all_results_temperature.py --config-path="../configs/comparison_baseline_cifar10lt" --config-name="default"
python scripts/compare_all_results_temperature.py --config-path="../configs/comparison_baseline_cifar100lt" --config-name="default"
"""
import pandas as pd
import hydra
from omegaconf import OmegaConf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from compare_results import process_experiment_logits
import yaml
import numpy as np

# Custom function to format mean and std


@hydra.main(config_path="../configs/comparison_baseline_cifar10lt", config_name="default", version_base=None)
def main(args):
  compare_results_losses(args)

def bold_max_min_value(val, col_name, col_max, col_min):
  # Function to format the maximum or min depending on metric
  if isinstance(val, (int, float)):
    if col_name in ('acc', 'f1') and val == col_max[col_name]:
      return '\\textbf{' + f'{val:.3f}' + '}'
    elif col_name in ('brier', 'ece', 'nll') and val == col_min[col_name]:
      return '\\textbf{' + f'{val:.3f}' + '}'
    return f'{val:.3f}'
  return str(val)


def bold_max_min_value_str(val, col_name, col_max, col_min):
  # Function to format the maximum or min depending on metric
  if "± " in val: #isinstance(val, (int, float)):
    if col_name in ('Acc.', 'F1', 'Cal-PR auc') and val == col_max[col_name]:
      return '\\textbf{' + f'{val}' + '}'
    elif col_name in ('Brier Score', 'ECE', 'NLL') and val == col_min[col_name]:
      return '\\textbf{' + f'{val}' + '}'
    return f'{val}'
  return str(val)


def bold_max_min_value_str_v1(val, col_name, col_max, col_min, group_max, group_min):
  # Function to format the maximum or min depending on metric
  if "± " in val: #isinstance(val, (int, float)):
    if col_name in ('Acc.', 'F1', 'Cal-PR auc') and val in group_max[col_name].values:
      if val == col_max[col_name]:
        return '\\textbf{\\textcolor{red}{' + f'{val}' + '}}'
      else:
        return '\\textbf{' + f'{val}' + '}'
    elif col_name in ('Brier Score', 'ECE', 'NLL') and val in group_min[col_name].values:
      if val == val == col_min[col_name]:
        return '\\textbf{\\textcolor{red}{' + f'{val}' + '}}'
      else:
        return '\\textbf{' + f'{val}' + '}'
    return f'{val}'
  return str(val)


def combine_mean_std(group):
  result = {}
  for metric in group.columns.levels[0]:
    mean_col = (metric, 'mean')
    var_col = (metric, 'std')

    mean_val = group[mean_col]
    std_val = group[var_col]
    if metric in ("Acc.", "F1"):
      mean_val *= 100
      std_val *= 100

    # only show mean
    combined = np.round(mean_val,3).astype(str) + ' (± ' + np.round(std_val,3).astype(str) + ')'
    combined = np.round(mean_val,3).astype(str) #+ ' (± ' + np.round(std_val,3).astype(str) + ')'
    result[metric] = combined
  return pd.DataFrame(result)


def combine_rows(group):
  results ={}
  for metric in group.columns:
    metric_val = group[metric]

    # for long tail data acc can change
    #if metric in ("Acc.", "F1"):
    #  metric_val = metric_val.split("/")[0]
    results[metric] = metric_val

  return pd.DataFrame(results)



def bold_max_min_value_str_dual(val, col_name, col_max, col_min):
  # Function to format the maximum or min depending on metric
  if "/" in val: #isinstance(val, (int, float)):
    val_ = val.split("/")[1]
    if col_name in ('Acc.', 'F1', 'Cal-PR auc') and val_ == col_max[col_name]:
      return '\\textbf{' + f'{val}' + '}'
    elif col_name in ('Brier Score', 'ECE', 'NLL') and val_ == col_min[col_name]:
      return '\\textbf{' + f'{val}' + '}'
    return f'{val}'
  return str(val)

def compare_results_losses(args):

  all_results = []

  for loss_type in args.train_loss:
    print(f"Loss type: {loss_type}")
    config_file = args.config_path + "/" + loss_type + ".yaml"
    loss_args = yaml.load(open(str(config_file)), Loader=yaml.FullLoader)
    loss_args = OmegaConf.create(loss_args)

    results = process_experiment_logits(loss_args)
    all_results.append(results)
    #break
  all_results = pd.concat(all_results)


  # Custom function to format mean and std
  def pandas_dataframe_to_bold_latex(df):
    # calculate mean and std, then format it
    # calculate mean, and add std to formatter function
    cols_to_group = ["data_type", "architecture"]

    df.index = df.index.droplevel(cols_to_group)
    # groupby mean and std

    df_new = df.groupby(df.index.names).agg([('mean', 'mean'), ('std', 'std')])
    df_new = df_new.groupby(level=0, axis=0).apply(combine_mean_std)

    # acc should not be recap?
    #df_new2 = df_new.groupby(["Train Loss","Ensemble Type"]).agg(lambda x: '/'.join(x.astype(str))if not(x.name in ('Acc.', 'F1')) else x[0])#.reset_index()
    df_new2 = df_new.groupby(["Train Loss","Ensemble Type"]).agg(lambda x: '/'.join(x.astype(str)))#.reset_index()
    # calculate col max and min
    col_max = df_new.max()
    col_min = df_new.min()

    # group_max and group_min, for group we care about
    group_min = df_new.groupby(['sdata_type','Train Loss']).min()
    group_min = group_min.loc['temperature']
    group_max = df_new.groupby(['sdata_type','Train Loss']).max()
    group_max = group_max.loc['temperature']

    col_min = group_min.min()
    col_max = group_max.max()

    #df = df.rename(index=index_mapping)
    columns = ['Acc.', 'F1', 'Brier Score', 'NLL', 'Cal-PR auc']
    #columns = ['acc', 'f1', 'brier', 'ece', 'nll']
    # slit col max and min
    formatters = {col: (lambda val, col_name=col: bold_max_min_value_str_dual(val, col_name, col_max, col_min)) for col in df_new2.columns}

    custom_order_train= {'Softmax (ERM)': 1,
     'Balanced Softmax CE': 2,
     "Weighted Softmax CE": 3,
     'd-Weighted Softmax CE': 4,
     }

    custom_order_ensemble = {'single model': 1,
      'avg. logits': 2,
      'avg. probs': 3,
      }
    # reset index temporarily
    df_reset = df_new2.reset_index()
    # sort the order
    df_reset['Train Loss'] = df_reset['Train Loss'].map(custom_order_train)
    df_reset['Ensemble Type'] = df_reset['Ensemble Type'].map(custom_order_ensemble)
    df_sorted = df_reset.sort_values(by=['Train Loss','Ensemble Type'])
    # change name again
    reorder_train = {value: key for key, value in custom_order_train.items()}
    reorder_etype = {value: key for key, value in custom_order_ensemble.items()}
    df_sorted['Train Loss'] = df_sorted['Train Loss'].replace(reorder_train)
    df_sorted['Ensemble Type'] = df_sorted['Ensemble Type'].replace(reorder_etype)

    # Set the MultiIndex back
    df_sorted.set_index(['Train Loss', 'Ensemble Type'], inplace=True)
    # change back to original name
    latex_str = df_sorted.to_latex(escape=False, formatters=formatters, index=True, columns=columns, multirow=True)
    return latex_str



  index_mapping = {
    'train_loss': 'Train Loss',
     'models': 'Ensemble Type',
     }


  all_results = all_results.rename_axis(index=index_mapping)

  index_mapping = {
    'acc': 'Acc.',
    'f1': 'F1',
    'brier': 'Brier Score',
    'ece': 'ECE',
    'nll': 'NLL',
    'CalPRauc' :'Cal-PR auc',
  }
  all_results = all_results.rename(columns=index_mapping)

  #%%
  all_results =  all_results.reset_index()
  all_results['Ensemble Type'] = all_results['Ensemble Type'].replace(
    {'no_avg': 'single model',
     'avg_logits': 'avg. logits',
     'avg_probs': 'avg. probs',
     })
  all_results['Train Loss'] = all_results['Train Loss'].replace(
    {'base': 'Softmax (ERM)',
     'base_bloss': 'Balanced Softmax CE',
     'weighted_softmax': "Weighted Softmax CE",
     'weighted_ce': 'd-Weighted Softmax CE',
     }
  )

  #import pdb; pdb.set_trace()
  all_results.set_index(['data_type','Train Loss',"sdata_type", 'Ensemble Type','architecture'], inplace=True)

  # check each ensemble before and after temperature scaling should have the same performance:
  #import pdb; pdb.set_trace()
  all_results
  # Plot
  cols_to_group_ = ["data_type", "architecture"]
  new_results = all_results.groupby(cols_to_group_)
  for group_name, group_data in new_results:
     print(group_name, flush=True)
     #group_data.drop(['data_type', 'sdata_type','architecture'], inplace=True)
     latex_output = pandas_dataframe_to_bold_latex(group_data)
     print(latex_output, flush=True)
     print("\n\n", flush=True)

  return all_results



if __name__ == "__main__":
    main()
