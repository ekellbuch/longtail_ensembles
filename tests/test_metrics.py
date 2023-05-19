from absl.testing import absltest
from absl.testing import parameterized
from longtail_ensembles.metrics import CalibrationAUC
import numpy as np


# Function to generate imbalanced datasets with varying positive class ratios
def generate_imbalanced_data(pos_ratio, num_samples=1000):
    num_positives = int(num_samples * pos_ratio)
    num_negatives = num_samples - num_positives
    y_true = np.concatenate([np.ones(num_positives), np.zeros(num_negatives)])
    y_proba = np.random.rand(num_samples, 2)
    return y_true, y_proba



class ModuleTestLoss(parameterized.TestCase):

  @parameterized.named_parameters(
    ('calib_roc_imb1', 0.1),
    ('calib_roc_imb3', 0.3),
    ('calib_roc_imb5', 0.5),
    ('calib_roc_imb7', 0.7),
  )
  def test_metrics_aucroc(self, pos_ratio):

    target, output = generate_imbalanced_data(pos_ratio)

    # calibration roc auc:
    # auc-roc value is the same across, but interpretation changes depending on correct_pred_as_pos_label
    # same value because we measure tradeoff to separate correct and incorrect predictions.
    # correct_pred_as_pos_label = True: curve is tradeoff for the presence of correct predictions.

    score = CalibrationAUC(curve='roc', correct_pred_as_pos_label=False)
    metric1 = score.auc(output, target)

    score = CalibrationAUC(curve='roc', correct_pred_as_pos_label=True)
    metric2 = score.auc(output, target)

    self.assertAlmostEqual(metric1, metric2, delta=1e-5)


  @parameterized.named_parameters(
    ('calib_pr_imb1', 0.1),
    ('calib_pr_imb3', 0.3),
    ('calib_pr_imb5', 0.5),
    ('calib_pr_imb7', 0.7),
  )
  def test_metrics_aucpr(self, pos_ratio):

    target, output = generate_imbalanced_data(pos_ratio)

    # calibration pr auc:
    # correctly classify positive instances w a low false positive rate.

    # correct_pred_as_pos_label=False:
    # recall and precision
    # recall is high when all the correct predictions are present and nothing is missed (no FN).
    # precision is high when there are no extra FP

    # if we set correct_pred_as_pos_label=True, 1: correct, 0 incorrect, model: p
    # TP: correct with p+
    # FN: correct with p-
    # TN: incorrect with p-
    # FP: incorrect with p+

    # TPR = TP/(TP+FN) : 1 = correct w p+. no correct w p-
    # FPR = FP/(FP+TN) : 0 = incorrect w p-
    # precision = TP/(TP+FP) : 1 = all correct w p+. no incorrect w p+.

    # TPR vs FPR: correct w p+, incorrect w p-
    # TPR vs precision: correct w p+, no incorrect w p+.

    # --------
    # if we set correct_pred_as_pos_label=False, 0: correct, 1: incorrect, model = 1-p
    # TP: incorrect with (1-p)+
    # FN: incorrect with (1-p)-
    # TN: correct with (1-p)-
    # FP: correct with (1-p)+

    # TPR = TP/(TP+FN) : 1 = no incorrect with (1-p)-
    #  proportion of actual incorrect predictions that model identifies as such.

    # FPR = FP/(FP+TN) : 0 =  correct with (1-p)-
    # precision = TP/(TP+FP) : 1 = incorrect with (1-p)+, no correct with (1-p)+
    #                           proportion of the incorrect predictions that are incorrect

    # TPR vs FPR: no incorrect with (1-p)-, and  correct w (1-p)-
    #            no incorrect with high prob, and correct w high prob.

    # TPR vs precision: no incorrect with (1-p)-, and no correct with (1-p)+
    #           no incorrect with high prob and no correct with low p.

    # identify false positive while minimizing false negatives.
    score = CalibrationAUC(curve='pr', correct_pred_as_pos_label=False)
    metric1 = score.auc(output, target)

    score = CalibrationAUC(curve='pr', correct_pred_as_pos_label=True)
    metric2 = score.auc(output, target)

    #self.assertGreaterEqual(metric2, metric1 )
    print(metric1, metric2)


if __name__ == '__main__':
  absltest.main()
