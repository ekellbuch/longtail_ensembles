"""
Calculate the metrics for different ensemble types

python scripts/compare_results.py --config-path="../results/configs/comparison_baseline_cifar10lt" --config-name="base"
"""
import hydra
from longtail_ensembles.utils_print import process_experiment_logits

@hydra.main(config_path="../results/configs/comparison_baseline_cifar10", config_name="base")
def main(args):
  process_experiment_logits(args)


if __name__ == "__main__":
    main()
