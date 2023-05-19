import pandas as pd
import numpy as np
from .utils_ens import build_ensemble, get_model_metrics_pclass, get_all_ensemble_pairs, get_model_metrics


def process_experiment_logits_pclass(args):
  results = []

  for ensemble_type_ in args.ensemble_types:
    for data_type in args.data_types:
      for model_architecture in args.models:
        model = args.models[model_architecture]
        all_new_model_pairs = get_all_ensemble_pairs(model)
        for ensemble_pair_idx, ensemble_pair in enumerate(all_new_model_pairs):
          for sdata_type in args.data_types[data_type]:
            file_ext = args.data_types[data_type][sdata_type]['logit']
            label_ext = args.data_types[data_type][sdata_type]['label']
            ens = build_ensemble(model=ensemble_pair,
                                 ensemble_method=ensemble_type_,
                                 file_ext=file_ext,
                                 label_ext=label_ext,
                                 )
            metrics = get_model_metrics_pclass(ens, modelname=ensemble_type_)
            metrics['data_type'] = data_type
            metrics['sdata_type'] = sdata_type
            metrics['architecture'] = model_architecture
            metrics['seed'] = ensemble_pair_idx

            results.append(metrics)

  results = pd.concat(results)
  results["models"] = pd.Categorical(results["models"])
  results["data_type"] = pd.Categorical(results["data_type"])
  results["sdata_type"] = pd.Categorical(results["sdata_type"])
  results["train_loss"] = args.train_loss
  results["train_loss"] = pd.Categorical(results["train_loss"])
  results["seed"] = pd.Categorical(results["seed"])
  results["architecture"] = pd.Categorical(results["architecture"])

  results.set_index(["data_type", "train_loss", "sdata_type","models","architecture"], inplace=True)

  return results


def process_experiment_logits(args):
  results = []

  for ensemble_type_ in args.ensemble_types:
    for data_type in args.data_types:  # ind vs ood
      for model_architecture in args.models:
        model = args.models[model_architecture]
        all_new_model_pairs = get_all_ensemble_pairs(model)
        for ensemble_pair_idx, ensemble_pair in enumerate(all_new_model_pairs):
          for sdata_type in args.data_types[data_type]:  # base and temperature
            file_ext = args.data_types[data_type][sdata_type]['logit']
            label_ext = args.data_types[data_type][sdata_type]['label']
            ens = build_ensemble(model=ensemble_pair,
                                 ensemble_method=ensemble_type_,
                                 file_ext=file_ext,
                                 label_ext=label_ext,
                                 )
            metrics = get_model_metrics(ens, modelname=ensemble_type_)
            metrics['data_type'] = data_type
            metrics['sdata_type'] = sdata_type
            metrics['architecture'] = model_architecture
            metrics['seed'] = ensemble_pair_idx
            results.append(metrics)

  results = pd.concat(results)
  results["models"] = pd.Categorical(results["models"])
  results["data_type"] = pd.Categorical(results["data_type"])
  results["sdata_type"] = pd.Categorical(results["sdata_type"])
  results["train_loss"] = args.train_loss
  results["train_loss"] = pd.Categorical(results["train_loss"])
  results["seed"] = pd.Categorical(results["seed"])
  results["architecture"] = pd.Categorical(results["architecture"])

  results.set_index(["data_type", "train_loss", "sdata_type", "models", "architecture"], inplace=True)

  # make sure the ensemble acc is the same before and after temperature scaling
  acc_condition = results.groupby(['data_type', 'train_loss', 'models', 'architecture', 'seed'])['acc'].unique()
  try:
    assert (acc_condition.apply(lambda x: np.all(np.isclose(x, x[0], atol=1e-2)))).all()
  except:
    # accuracy should not change:
    pd.set_option('display.max_rows', 500)
    acc_ineq = results.groupby(['data_type', 'train_loss', 'models', 'architecture', 'seed'])['acc'].apply(lambda x: x[1] >= x[0])
    print(acc_ineq)
  # make sure the ensemble nll is the lower before/ after temperature scaling
  for metric_ in ['nll', 'ece']:
    nll_ = results.groupby(['data_type', 'train_loss', 'models', 'architecture', 'seed']).apply(lambda x: x.groupby(['sdata_type'])[metric_].mean())
    try:
      assert (nll_['temperature'] <= nll_['base']).all()
    except:
      pd.set_option('display.max_rows', 500)
      nll_ = results.groupby(['data_type', 'train_loss', 'models', 'architecture', 'seed']).apply(lambda x: x.groupby(['sdata_type'])[metric_].unique())

      print('Failed for metric {}'.format(metric_), flush=True)
      print(nll_, flush=True)

  print(results, '\n\n', flush=True)
  print('passed all checks', flush=True)


  return results