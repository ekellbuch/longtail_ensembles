"""
Given a yaml file with locations of logits,
Calculate the model performance for a variety of models and store outputs in csv file.

python scripts/calculate_model_performance.py --config-path="../results/configs/datasets/cifar10" --config-name="cifar10"
python scripts/calculate_model_performance.py --config-path="../results/configs/datasets/cifar10" --config-name="cinic10"
python scripts/calculate_model_performance.py --config-path="../results/configs/datasets/cifar10" --config-name="cifar10_1"
python scripts/calculate_model_performance.py --config-path="../results/configs/datasets/imagenet" --config-name="imagenet_c_gaussian_noise_1"
"""
import hydra
from ensemble_testbed.predictions import Model
import os
import pandas as pd
from pathlib import Path

here = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


def get_arrays_toplot(models):
    """
  Takes as argument a dictionary of models:
  keys giving model names, values are dictionaries with paths to individual entries.
  :param models: names of individual models.
  """
    all_metrics = []
    for modelname, model in models.items():
        num_models = len(model.filepaths)
        hparams = model.hyperparameters
        if not(len(hparams) == num_models):
            hparams = [hparams] * num_models
        for ii in range(num_models):
            m = model.filepaths[ii]
            l = model.labelpaths[ii]
            h = hparams[ii]
            emodel = Model(m, "ind")
            npz_flag = model.get('npz_flag', None)

            emodel.register(
                filename=m,
                inputtype=None,
                labelpath=l,
                logits=True,
                npz_flag=npz_flag,
            )
            acc, nll, brier, qunc \
              = emodel.get_accuracy(), emodel.get_nll(), emodel.get_brier(), emodel.get_qunc()
            print("{}-{}: Acc: {:.3f}, NLL: {:.3f}, Brier: {:.3f} Qunc:{:.3f}".
                  format(modelname, ii, acc, nll, brier, qunc))
            all_metrics.append([modelname, m, l, acc, nll, brier, qunc, h, ii, npz_flag])

    df = pd.DataFrame(all_metrics,
                      columns=[
                          "models", "filepaths", "labelpaths", "acc", "nll",
                          "brier", "qunc", "hyperparameters", "seed", "npz_flag"
                      ])
    return df


@hydra.main(config_path="../results/configs/datasets/cifar10", config_name="cifar10")
def main(args):
    results_dir = Path(here) / "results/model_performance/{}.csv".format(
        args.test_set)
    os.makedirs(os.path.dirname(results_dir), exist_ok=True)

    print('\n dataset {} results_dir: {}\n'.format(args.title, results_dir))

    # Get performance metrics for each ensemble:
    df = get_arrays_toplot(args.models)

    # additional info:
    df['train_set'] = args.train_set
    df['test_set'] = args.test_set
    df['model_family'] = args.model_family

    # Dump to csv
    df.to_csv(str(results_dir))

    print('Stored performance of {} in {}'.format(args.test_set, results_dir))

    return


if __name__ == "__main__":
    main()
