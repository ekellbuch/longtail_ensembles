from absl.testing import absltest
from absl.testing import parameterized
from longtail_ensembles.metrics import CalibrationAUC
import numpy as np
from longtail_ensembles.utils_ens import build_ensemble, get_model_metrics,get_all_ensemble_pairs


# Function to generate imbalanced datasets with varying positive class ratios
def generate_imbalanced_data(pos_ratio, num_samples=1000):
    num_positives = int(num_samples * pos_ratio)
    num_negatives = num_samples - num_positives
    y_true = np.concatenate([np.ones(num_positives), np.zeros(num_negatives)])
    y_proba = np.random.rand(num_samples, 2)
    return y_true, y_proba

CONFIG = {
  "cifar10lt": "/data/Projects/linear_ensembles/longtail_ensembles/configs/comparison_baseline_cifar10lt/"
}

def read_dataset_yaml(ind_dataset):




class ModuleTestEnsembles(parameterized.TestCase):
  @parameterized.named_parameters(
    ('logit_ens', 'avg_logits', 2),
    ('prob_ens', 'avg_probs', 2),
  )
  def test_metrics_aucroc(self, ensemble_method, num_models):


    total_samples = 100
    num_classes = 2
    output = np.randn(num_models, total_samples, num_classes)
    target = np.randint(0, num_classes, (total_samples,))

    model
    all_new_model_pairs = get_all_ensemble_pairs(model)
    for ensemble_pair_idx, ensemble_pair in enumerate(all_new_model_pairs):
      ens = build_ensemble(model=ensemble_pair,
                           ensemble_method=ensemble_type_,
                           file_ext=file_ext,
                           label_ext=label_ext,
                           )
      metrics = get_model_metrics(ens, modelname=ensemble_type_)


    build_ensemble()

