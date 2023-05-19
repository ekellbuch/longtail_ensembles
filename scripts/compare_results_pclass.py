"""
Given a config
calculate the metrics for different ensemble types

python scripts/compare_results_pclass.py --config-path="../configs/comparison_baseline_cifar10lt" --config-name="base"
"""
import hydra
from longtail_ensembles.utils_print import process_experiment_logits_pclass

@hydra.main(config_path="../configs/comparison_baseline_cifar10", config_name="base", version_base=None)
def main(args):
  process_experiment_logits_pclass(args)


if __name__ == "__main__":
    main()
