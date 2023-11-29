"""
calculate the metrics for different ensemble types

python scripts/compare_all_results.py --config-path="../configs/results/comparison_baseline_cifar10lt" --config-name="default"
python scripts/compare_all_results.py --config-path="../configs/results/comparison_baseline_cifar100lt" --config-name="default"
"""
import pandas as pd
import hydra
from omegaconf import OmegaConf
from longtail_ensembles.utils_print import process_experiment_logits
import yaml
import numpy as np

@hydra.main(config_path="../results/configs/comparison_baseline_cifar10lt", config_name="default", version_base=None)
def main(args):
  compare_results_losses(args)


def bold_max_min_value_str(val, col_name, col_max, col_min):
  # Function to format the maximum or min depending on metric:
  if "± " in val:
    if col_name in ('Acc.', 'F1', 'Cal-PR auc') and val == col_max[col_name]:
      return '\\textbf{' + f'{val}' + '}'
    elif col_name in ('Brier Score', 'ECE', 'NLL') and val == col_min[col_name]:
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

    combined = np.round(mean_val,3).astype(str) + ' (± ' + np.round(std_val,3).astype(str) + ')'
    result[metric] = combined
  return pd.DataFrame(result)


def compare_results_losses(args):

  all_results = []

  for loss_type in args.train_loss:
    print(f"Loss type: {loss_type}")
    config_file = args.config_path + "/" + loss_type + ".yaml"
    loss_args = yaml.load(open(str(config_file)), Loader=yaml.FullLoader)
    loss_args = OmegaConf.create(loss_args)

    results = process_experiment_logits(loss_args)
    all_results.append(results)
  all_results = pd.concat(all_results)

  cols_to_group = ["data_type", "sdata_type", "architecture"]

  # Custom function to format mean and std in printed table
  def pandas_dataframe_to_bold_latex(df):
    # calculate mean and std, then format it
    df.index = df.index.droplevel(cols_to_group)
    # groupby mean and std

    df_new = df.groupby(df.index.names).agg([('mean', 'mean'), ('std', 'std')])
    df_new = df_new.groupby(level=0, axis=0).apply(combine_mean_std)

    # calculate col max and min
    col_max = df_new.max()
    col_min = df_new.min()

    columns = ['Acc.', 'F1', 'Brier Score', 'NLL', 'Cal-PR auc']
    formatters = {col: (lambda val, col_name=col: bold_max_min_value_str(val, col_name, col_max, col_min)) for col in df_new.columns}

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
    df_reset = df_new.reset_index()
    # sort the order
    df_reset['Train Loss'] = df_reset['Train Loss'].map(custom_order_train)
    df_reset['Ensemble Type'] = df_reset['Ensemble Type'].map(custom_order_ensemble)
    df_sorted = df_reset.sort_values(by=['Train Loss', 'Ensemble Type'])
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

  all_results.set_index(['data_type','Train Loss','sdata_type','Ensemble Type','architecture'], inplace=True)

  # make sure ensemble accuracy is the same before and after temperature scaling for no_avg model
  # when ensembling temperature scaled models, the performance can change.
  try:
    assert (all_results.groupby(['data_type', 'Train Loss', 'Ensemble Type', 'architecture', 'seed'])['Acc.'].nunique().loc[pd.IndexSlice[:,:,"single model"]] == 1).all()
  except:
    pd.set_option('display.max_rows', 500)
    print(all_results.groupby(['data_type', 'Train Loss', 'Ensemble Type', 'architecture','seed'])['Acc.'].unique(), flush=True)
    exit()
  # make sure the ensemble nll is the lower before/ after temperature scaling
  for metric_ in ['NLL', 'ECE']:
    nll_ = all_results.groupby(['data_type', 'Train Loss', 'Ensemble Type', 'architecture', 'seed']).apply(lambda x: x.groupby(['sdata_type'])[metric_].mean())
    try:
      assert (nll_['temperature'] < nll_['base']).loc[pd.IndexSlice[:,:,"single model"]].all()
    except:
      pd.set_option('display.max_rows', 500)
      print(nll_, flush=True)
      exit()

  new_results = all_results.groupby(cols_to_group)
  for group_name, group_data in new_results:
     print(group_name, flush=True)
     latex_output = pandas_dataframe_to_bold_latex(group_data)
     print(latex_output, flush=True)
     print("\n\n", flush=True)

  return all_results



if __name__ == "__main__":
    main()
