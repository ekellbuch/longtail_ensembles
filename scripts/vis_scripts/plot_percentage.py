"""
Find percentage of models better than one another for each metric
Run:
  python scripts/vis_scripts/plot_percentage.py --ind_dataset=cifar10 --ood_dataset=cinic10
"""

#%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import fire
import seaborn as sns
from longtail_ensembles.utils import read_dataset, pretty_data_names
from scipy import stats

BASE_DIR = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))).parent
#BASE_DIR = Path("/data/Projects/linear_ensembles/longtail_ensembles")
print(BASE_DIR)
plt.style.use(os.path.join(str(BASE_DIR), "scripts/vis_scripts/stylesheet.mplstyle"))

output_dir = BASE_DIR / "results"

#%

#%%

def main(ind_dataset='cifar10', ood_dataset="cinic10", out_name="model_performance/het_run_v6"):

  #%%
  #ind_dataset = 'cifar10'
  #ood_dataset = 'cinic10'
  ##ind_dataset = 'imagenet'
  #ood_dataset = 'imagenetv2mf'
  #out_name = "model_performance/het_run_v6"

  #%%
  data_filename = output_dir / out_name / "{}.csv".format(ind_dataset)
  all_datas = read_dataset(data_filename)
  data_filename = output_dir / out_name / "{}.csv".format(ood_dataset)
  ood_datas = read_dataset(data_filename)
  all_datas = pd.concat([all_datas, ood_datas], axis=0)

  output_filename = output_dir / "figures" / out_name / ("ens_th_{}_{}.pdf".format(ind_dataset, ood_dataset))

  os.makedirs(os.path.dirname(output_filename), exist_ok=True)
  print('Storing output in:\n{}'.format(output_filename), flush=True)

  all_subdatas=all_datas.copy()
  all_subdatas = all_subdatas[np.logical_or(all_subdatas['dataset']==ind_dataset,all_subdatas['dataset']==ood_dataset)]
  ens_size = all_subdatas['ensemble_size'].values
  ens_size = np.unique(ens_size)
  assert len(ens_size) == 1

  all_subdatas['0-1 Error'] = 1 - all_subdatas['acc']
  all_subdatas.rename(columns={'nll': 'NLL',
                               'ece':'ECE',
                               'brier':'Brier',
                               'f1':'F1'}, inplace=True)
  all_subdatas.rename(columns={'ensemble_avg_type': 'Ensemble Type'}, inplace=True)
  all_subdatas['Ensemble Type'] = all_subdatas['Ensemble Type'].replace(
    {'avg_logits': 'avg. logits',
     'avg_probs': 'avg. probs',
     'no_avg': 'single model'})

  #all_subdatas = all_subdatas[all_subdatas['acc'] >= 0.25]
  all_subdatas = all_subdatas[~all_subdatas['models'].str.contains('deepens')]
  plot_results = all_subdatas.copy()
  plot_results["data_type"] = plot_results["data_type"].replace({"ind": "InD", "ood": "OOD"})
  plot_results['Ensemble Type'] = pd.Categorical(plot_results['Ensemble Type'])
  plot_results['dataset'] = pd.Categorical(plot_results['dataset'])
  plot_results['data_type'] = pd.Categorical(plot_results['data_type'])
  #%
  plot_results = plot_results.drop(columns=['model_scores',
                                            'ensemble_size',
                                            'ensemble_type',
                                            'architecture',
                                            'dataset',
                                            'seed',
                                            'num_params',
                                            'binning',
                                            'acc',
                                            'ensemble_group',
                                            ])
  plot_results.drop_duplicates(inplace=True)
  plot_results.rename(columns={'data_type': 'Data Type'},inplace=True)
  metrics_columns=["0-1 Error", "NLL", "Brier","F1", "ECE", "CalROCauc", "CalPRauc","EnsVar","ClsVar"]
  #metrics_columns=["EnsVar","ClsVar"]
  # does this sort ind and ood separately?
  #plot_results = plot_results.sort_values(by="0-1 Error")
  plot_results = plot_results.reset_index(drop=True)

  new_plot_results = pd.melt(plot_results, id_vars=['models', 'Data Type', 'Ensemble Type'], value_vars=metrics_columns,
                      var_name='Metric', value_name='Value')
  new_plot_results = new_plot_results.dropna()

  model_types =['single model', 'avg. logits', 'avg. probs']
  colors = ['yellowgreen', 'tomato', 'dodgerblue', 'violet']

  #%%
  metrics_columns=["0-1 Error", "F1", "NLL", "Brier","CalROCauc", "CalPRauc"]
  #metrics_columns=["EnsVar","ClsVar"]
  #metrics_columns=["CalPRauc"]

  new_plot_results = new_plot_results[new_plot_results['Ensemble Type'] != 'single model']

  #%%
  # probability better marginals
  all_outs = []
  for metric in metrics_columns:
    for data_type in ["InD", "OOD"]:

        plot_metric_outs = new_plot_results[new_plot_results['Metric'] == metric]
        plot_metric_outs = plot_metric_outs[plot_metric_outs['Data Type'] == data_type]

        # h0: null hypothesis, there is no statistical difference between the two methods
        # h1: opposite of h0
        rvs1 = plot_metric_outs[plot_metric_outs['Ensemble Type'] == 'avg. logits']['Value'].values
        rvs2 = plot_metric_outs[plot_metric_outs['Ensemble Type'] == 'avg. probs']['Value'].values
        # independent samples (only means)
        #t_statistic, p_value = stats.ttest_ind(rvs1, rvs2, equal_var=True)
        # because we are comparing related samples.
        # assumes equal variance:
        t_statistic, p_value = stats.ttest_rel(rvs1, rvs2)
        # assumes equal variance:
        t_statistic, p_value = stats.wilcoxon(rvs1, rvs2)
        # pair wise mse
        mse_ = np.mean((rvs1 - rvs2)**2)
        se_max = np.max((rvs1 - rvs2)**2)
        se_min = np.min((rvs1 - rvs2)**2)
        mae_ = np.mean(np.abs(rvs1 - rvs2))
        assert  len(rvs1) == len(rvs2)
        #print(t_statistic, p_value)
        alpha = 0.05
        if p_value < alpha:
          #print("Reject the null hypothesis. Methods are statistically different.")
          stat_diff = True
        else:
          #print("Fail to reject the null hypothesis. No significant difference.")
          stat_diff = False

        if mse_ >= 1e-3:
          pract_sig = True
        else:
          pract_sig = False

        outs = {}
        outs['Num models'] = len(rvs1)
        outs['metric'] = metric
        outs['dataset'] = ind_dataset if data_type == 'InD' else ood_dataset
        outs['mse'] = mse_
        #outs['se_min'] = se_min
        outs['se_max'] = se_max
        outs['stat_diff'] = stat_diff
        """
        outs['mae'] = mae_
        #outs['t_statistic'] = t_statistic
        outs['p value'] = p_value
        outs['data_type'] = data_type
        outs['pract_sig'] = pract_sig
        """
        all_outs.append(outs)

  def latex_format(val):
    if val is True:
      return r'\textbf{True}'
    else:
      return str(val)

  all_outs = pd.DataFrame(all_outs)
  print(all_outs)

  all_outs['dataset'] = all_outs['dataset'].replace(pretty_data_names)
  all_outs = all_outs.drop(columns=['Num models'])

  all_outs.set_index(['dataset', 'metric'], inplace=True)
  all_outs.sort_values(by='dataset', inplace=True)

  #float_format = "%.3e"
  float_format = "{:.5f}".format
  formatters = {}
  #all_outs['stat_diff'] = all_outs['stat_diff'].apply(latex_format)
  #all_outs['pract_sig'] = all_outs['pract_sig'].apply(latex_format)
  latex_str = all_outs.to_latex(escape=False, formatters=formatters, index=True, float_format=float_format,)
  # boldface
  print(latex_str)

  #%%
  return


#%%
if __name__ == "__main__":
    fire.Fire(main)
