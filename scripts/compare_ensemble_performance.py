"""
Form an ensemble and evaluate its performance

"""
from longtail_ensembles.utils_ens import build_ensemble, get_model_metrics
import pandas as pd
import hydra


@hydra.main(config_path="../configs/deprecated/base_temperature", config_name="cifar100__base__temperature", version_base=None)
def main(args):

  ensemble_types = ['no_avg', 'avg_logits', 'avg_probs']


  all_metrics = []
  for ensemble_type_ in ensemble_types:
    print('Build ensemble')
    for model_architecture in args.models:
      model = args.models[model_architecture]
      ens = build_ensemble(model=model,
                          ensemble_method=ensemble_type_,
                           )
      metrics = get_model_metrics(ens,
                                  modelname=ensemble_type_)

      all_metrics.append(metrics)

  print(pd.concat(all_metrics))


if __name__ == "__main__":
    main()
