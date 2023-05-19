"""
Tools to help calculate calibration related metrics given a predictions and labels.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, auc, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve
from collections import defaultdict

class BrierScoreData(object):
    """Calculates brier score.

  """

    def __init__(self):
        pass

    def brierscore(self, prob, target):
        """Given an array of probabilities, `prob`, (batch,dim) and an array of targets `target` (dim),
         calculates the brier score as if we were in the binary case:
         take the predicted probability of the target class, and just calculate based on that.
    :param prob: array of probabilities per class.
    :param target: list/array of true targets.

    """
        probs = prob[np.arange(len(target)), target]
        deviance = probs - np.ones(probs.shape)

        return np.mean(deviance**2)

    def brierscore_multi(self, prob, target):
        """The "original" brier score definition that accounts for other classes explicitly.
        Note the range of this test is 0-2.
    :param prob: array of probabilities per class.
    :param target: list/array of true targets.

    """
        target_onehot = np.zeros(prob.shape)
        target_onehot[np.arange(len(target)), target] = 1  ## onehot encoding.
        deviance = prob - target_onehot
        return np.mean(np.sum(deviance**2, axis=1))

    def brierscore_multi_vec(self, prob, target):
        """The "original" brier score definition that accounts for other classes explicitly.
        Note the range of this test is 0-2. output the vector per sample
    :param prob: array of probabilities per class.
    :param target: list/array of true targets.

    """
        target_onehot = np.zeros(prob.shape)
        target_onehot[np.arange(len(target)), target] = 1  ## onehot encoding.
        deviance = prob - target_onehot
        return np.sum(deviance**2, axis=1)


class VarianceData(object):
    """Calculates variance/related metrics. This is the variance in the confidence of the top predicted label.

  """

    def __init__(self):
        pass

    def dkl_qdag_qi(self, logits, target, per_class=True):
        """
        E_D [1_M \sum_{i=1^M} D_KL (\dag q || q_i)]
        where q_i is the prediction of model i
        M is the number of models and
        \dag q is the arithmetic mean prediction of the modelsd
        q_i are the model predictions
        """
        num_models, num_samples, num_classes = logits.shape
        # q_i are the individual model predictions:
        q_i = np.exp(logits)/ np.exp(logits).sum(axis=-1, keepdims=True)

        # dag_q = average the model probabilities
        bar_q = np.mean(q_i, axis=0)  # (samples x classes)

        D_KL = bar_q * np.log(bar_q / q_i)
        D_KL = np.sum(D_KL, axis=-1)  # sum along the classes

        # average along models
        D_KL = np.mean(D_KL, axis=0)

        if per_class == False:
            # average along samples
            return np.mean(D_KL)

        # average along samples, for each true class
        score = {}
        all_class_keys = np.arange(num_classes)
        for class_key in all_class_keys:
            mask_ = target == class_key
            # average along samples in class
            score[class_key] = np.mean(D_KL[mask_])
        return score

    def dkl_qbar_qi(self, logits, target, per_class=True):
        """
        E_D [1_M \sum_{i=1^M} D_KL (\bar q || q_i)]
        where q_i is the prediction of model i
        M is the number of models and
        \bar q is the normalized geometric prediction of the models
        q_i are the model predictions
        """
        num_models, num_samples, num_classes = logits.shape
        # q_i are the individual model predictions:
        q_i = np.exp(logits)/ np.exp(logits).sum(axis=-1, keepdims=True)

        # qbar: logit ensemble samples x classes
        # average the model logits and then apply a softmax
        bar_q = np.mean(logits, axis=0)
        bar_q = np.exp(bar_q)/ np.exp(bar_q).sum(axis=-1, keepdims=True)

        D_KL = bar_q * np.log(bar_q / q_i)
        D_KL = np.sum(D_KL, axis=-1)  # sum along the classes

        # average along models:
        D_KL = np.mean(D_KL, axis=0)

        if per_class == False:
            # average along samples
            return np.mean(D_KL)

        # average along samples, for each true class
        score = {}
        all_class_keys = np.arange(num_classes)
        for class_key in all_class_keys:
            mask_ = target == class_key
            # average along samples in class
            score[class_key] = np.mean(D_KL[mask_])
        return score

    def dkl_uni_eq(self, logits, target, per_class=True):
        """
        E_D [1_M \sum_{i=1^M} [log 1/M - log [q_i^{y}/ \sum_{j=1}^M q_j^{y}]] )]
        where q_i is the prediction of model i
        M is the number of models and
        depends on q_i not
        """
        num_models, num_samples, num_classes = logits.shape
        # q_i are the individual model predictions:
        q_i = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        # q_i are the individual model predictions for the true class:

        q_i_y = q_i[:, np.arange(num_samples), target]
        assert q_i_y.shape == (num_models, num_samples)

        log_1M = np.log(1/num_models)
        log_avg = np.log(q_i_y/q_i_y.sum(0, keepdims=True))

        D_KL = log_1M - log_avg

        # average along models
        D_KL = np.mean(D_KL, axis=0)

        # average along samples
        if per_class == False:
            return np.mean(D_KL)

        # average along samples, for each true class
        score = {}
        all_class_keys = np.arange(num_classes)
        for class_key in all_class_keys:
            mask_ = target == class_key
            # average along samples in class
            score[class_key] = np.mean(D_KL[mask_])
        return score

class AccuracyData(object):
    """Calculates accuracy related metrics.

  """

    def __init__(self):
        pass

    def accuracy(self, prob, target):
        """Given predictions (example,class) and targets (class), will calculate the accuracy.

    """
        selected = np.argmax(prob, axis=1)
        correct = target == selected
        accuracy = sum(correct) / len(target)
        return accuracy

    def accuracy_per_class(self, prob, target, num_classes=None):
        """
    Calculate per-class accuracy for a multi-class classification problem.

    Args:
        y_true (list or numpy array): List of true labels.
        y_pred (list or numpy array): List of predicted labels.

    Returns:
        dict: A dictionary containing per-class accuracy.
    """
        class_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})

        y_pred = np.argmax(prob, axis=1)

        for true_label, pred_label in zip(target, y_pred):
            class_accuracy[true_label]['total'] += 1
            if true_label == pred_label:
                class_accuracy[true_label]['correct'] += 1

        per_class_accuracy = {}
        for class_label, stats in class_accuracy.items():
            per_class_accuracy[class_label] = stats['correct'] / stats['total']

        if num_classes is not None:
            all_class_keys = np.arange(num_classes)
            local_class_keys = np.array(list(per_class_accuracy.keys()))
            extra_keys = np.setdiff1d(all_class_keys, local_class_keys)
            for class_label in extra_keys:
                per_class_accuracy[class_label] = 0.0
            assert len(per_class_accuracy) == num_classes

        return per_class_accuracy


class CalibrationData(object):

    def __init__(self, power=1, num_bins=15):
        self.num_bins = num_bins
        self.power = power

    def ece_todo(self, prob, target):
        preds = np.argmax(prob, -1)
        fs = np.max(prob, -1)
        ys = (preds == target).astype('float')
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # Expand
        fs = fs[..., None]
        ys = ys[..., None]

        # Get mask for samples in bin
        in_bin = (np.greater_equal(fs, bin_lowers) *
                  np.less(fs, bin_uppers)).transpose(-1, -2)
        tot_in_bin = np.clip(in_bin.sum(axis=-1), 1, None)  #.clip(1., None)
        #tot_in_bin = in_bin.sum(axis=-1) #.clip(1., None)
        prob_in_bin = (in_bin @ fs).squeeze(-1)
        true_in_bin = (in_bin @ ys).squeeze(-1)
        esces = (prob_in_bin - true_in_bin) / np.abs(tot_in_bin)**(self.power)

        return np.multiply(esces, tot_in_bin / ys.shape[0]).sum(-1)

    def ece(self, prob, target):
        """Calculates expected calibration error"""
        predictions = np.argmax(prob, 1)

        confidences = np.max(prob, 1)
        accuracies = predictions == target

        # First define the bins in the range (0, 1]
        bins = np.linspace(0, 1, self.num_bins + 1)
        bins[-1] = 1.0001
        # recall confidence = max_z p(z|x)
        # in this case we use the max value of p(z|x)
        confidences_idx = np.digitize(confidences, bins) - 1
        #assert confidences_idx.min() > 0
        #assert confidences_idx.max() <= num_bins - 1

        # the average accuracy per bin
        # Accuracy = (TP+TN)/(TP+TN+FP+FN)
        accuracy_bin = np.zeros(self.num_bins)
        # The average confidence per bin
        # confidence =  argmax_z p(z|x)
        confidence_bin = np.zeros(self.num_bins)
        # Expected calibration error
        # average (expected accuracy - expected confidence)* (TP+TN)/num_samples
        ece = 0

        # For each bin:
        for bi in range(self.num_bins):
            # mask of confidence values "bi" in bin
            in_bin = confidences_idx == bi
            # if there are any values in the bin
            if in_bin.sum() > 0:
                # set of indices of samples with prediction confidences in the
                # interval Im = ((m-1)/M, m/M] for m in M
                # bm = np.argwhere(in_bin).flatten()

                # (len(bm)/T) = (TP+TN)/T, where T is the total # samples
                prop_in_bin = in_bin.mean()

                # accuracy = 1/|Bm| sum_i in BM 1(hat_y_i = y_i)
                avg_accuracy_in_bin = accuracies[in_bin].mean()

                # confidence = 1/|Bm| sum_i in BM p_im
                avg_confidence_in_bin = confidences[in_bin].mean()

                # Expected calibration error
                ece += np.abs(avg_accuracy_in_bin -
                              avg_confidence_in_bin) * prop_in_bin

                # store vals
                accuracy_bin[bi] = avg_accuracy_in_bin
                confidence_bin[bi] = avg_confidence_in_bin
        return ece


class F1ScoreData(object):
    """Calculates accuracy related metrics.

  """

    def __init__(self):
        pass

    def f1_score(self, prob, target, average='macro'):
        """Given predictions (example,class) and targets (class), will calculate the accuracy.

    """
        selected = np.argmax(prob, axis=1)
        score_ = f1_score(target, selected, average=average)
        return score_
    def f1_score_per_class(self, prob, target, num_classes=None):
        """Given predictions (example,class) and targets (class), will calculate the accuracy.

    """
        selected = np.argmax(prob, axis=1)
        score_ = f1_score(target, selected, average=None)


        per_class_f1_score = {}
        for class_label, stats in enumerate(score_):
            per_class_f1_score[class_label] = stats

        if num_classes is not None:
            all_class_keys = np.arange(num_classes)
            local_class_keys = np.array(list(per_class_f1_score.keys()))
            extra_keys = np.setdiff1d(all_class_keys, local_class_keys)
            for class_label in extra_keys:
                per_class_f1_score[class_label] = 0.0
            assert len(per_class_f1_score) == num_classes

        return per_class_f1_score

class PrecisionScoreData(object):
    """Calculates precision related metrics.

  """

    def __init__(self):
        pass

    def precision_score(self, prob, target, average='macro'):
        """Given predictions (example,class) and targets (class), will calculate the accuracy.

    """
        selected = np.argmax(prob, axis=1)
        score_ = precision_score(target, selected, average=average)
        return score_


class RecallScoreData(object):
    """Calculates precision related metrics.

  """

    def __init__(self):
        pass

    def recall_score(self, prob, target, average='macro'):
        """Given predictions (example,class) and targets (class), will calculate the accuracy.

    """
        selected = np.argmax(prob, axis=1)
        score_ = recall_score(target, selected, average=average)
        return score_


class NLLData(object):
    """Calculates the negative log likelihood of the data.

  """

    def __init__(self):
        pass

    def nll(self, prob, target, normalize=False):
        """Given predictions (example,class) and targets (class), will calculate the negative log likelihood.
        Important here that the probs are expected to be outputs of softmax functions.

    """
        probs = prob[np.arange(len(target)), target]
        logprobs = np.log(probs)

        nll = -sum(logprobs)
        if normalize:
            nll = nll / len(logprobs)
        return nll

    def nll_vec(self, prob, target):
        """NLL contribution for each individual datapoint.

    """
        probs = prob[np.arange(len(target)), target]
        logprobs = np.log(probs)

        return -logprobs


class CalibrationDataV0(object):
    """Initializes an object to bin predictions that are fed to it in batches.

  """

    def __init__(self, binedges):
        """Initialize with a set of floats giving the interval spacing between different bins

    :param binedges: list of edges of the bins, not including 0 and 1.
    Will create intervals like `[[0,binedges[0]),[binedges[0],binedges[1]),...,[binedges[-1],100]]`
    """
        assert binedges[0] > 0 and binedges[
            -1] < 1, "bin edges must be strictly within limits."
        assert np.all(np.diff(binedges) > 0), "bin edges must be ordered"
        assert type(binedges) == list
        padded = [0] + binedges + [1]
        self.binedges = [(padded[i], padded[i + 1])
                         for i in range(len(padded) - 1)]

    def bin(self, prob, target):
        """Given predictions  (example, class) and targets  (class), will bin them according to the binedges parameter.
    Returns a dictionary with keys giving bin intervals, and values another dictionary giving the accuracy,
    confidence, and number of examples in the bin.
    """
        data = self.analyze_batch(prob, target)
        ## first let's divide the data by bin:
        bininds = np.array(list(data["bin"]))
        bin_assigns = [
            np.where(bininds == i) for i in range(len(self.binedges))
        ]
        ## now we want per-bin stats:
        all_stats = {}
        for ai, assignments in enumerate(bin_assigns):
            bin_card = len(assignments[0])
            name = self.binedges[ai]
            if bin_card == 0:
                bin_conf = np.nan
                bin_acc = np.nan
            else:
                bin_conf = sum(data["maxprob"][assignments]) / bin_card
                bin_acc = sum(data["correct"][assignments]) / bin_card
            all_stats[name] = {
                "bin_card": bin_card,
                "bin_conf": bin_conf,
                "bin_acc": bin_acc
            }
        return all_stats

    def ece(self, prob, target):
        """Calculate the expected calibration error across bins given a probability and target.

    """
        all_stats = self.bin(prob, target)
        ece_nonnorm = 0
        for interval, intervalstats in all_stats.items():
            if intervalstats["bin_card"] == 0:
                continue
            else:
                factor = intervalstats["bin_card"] * abs(
                    intervalstats["bin_acc"] - intervalstats["bin_conf"])
                ece_nonnorm += factor
        ece = ece_nonnorm / len(target)
        return ece

    def getbin(self, data):
        """Halt and return the index where your maximum prediction fits into a bin.

    """
        index = len(self.binedges) - 1
        for b in self.binedges[::-1]:  ## iterate in reverse order
            if data >= b[0]:
                break
            else:
                index -= 1
        return index

    def analyze_batch(self, prob, target):
        """Given a matrix of class probabilities (batch, class) and a target (class),
        returns calibration related info about that datapoint:
    {"prob":prob,"target":target,"correct":bool,"bin":self.binedges[index]}

    """
        assert len(prob.shape) == 2
        assert prob.shape[0] == len(target)
        maxprob, maxind = np.amax(prob, axis=1), np.argmax(prob, axis=1)
        correct = maxind == target
        binind = map(self.getbin, maxprob)
        return {
            "maxprob": maxprob,
            "maxind": maxind,
            "target": target,
            "correct": correct,
            "bin": binind
        }

class CalibrationAUC(object):
    """Calibration AUC-ROC
    Compute the AUC for a binary prediction task, where
    binary label is the predictive correctness (or incorrectness)
    and the prediction score is the confidence score (or 1-confidence score).
    See https://github.com/google/uncertainty-baselines/blob/master/baselines/toxic_comments/metrics.py
    curve: roc_curve,
  """
    def __init__(self, curve="roc", correct_pred_as_pos_label=True):
        self.correct_pred_as_pos_label = correct_pred_as_pos_label
        self.curve = curve

    def auc(self, prob, target):
        # binary_label is the predicted correctness of the prediction
        selected = np.argmax(prob, axis=1)
        binary_label = target == selected
        prediction_score = np.max(prob, axis=1)

        if not self.correct_pred_as_pos_label:
            # Use incorrect prediction as the positive class.
            # This is important since an accurate model has few incorrect predictions.
            # This results in label imbalance in the calibration AUC computation, and
            # can lead to overly optimistic results.
            prediction_score = 1. - prediction_score
            binary_label = 1. - binary_label

        # roc curve
        if self.curve == "roc":
            # fpr, tpr, th
            x, y, th = roc_curve(binary_label, prediction_score)
        # precision recall curve
        elif self.curve == "pr":
            # precision, recall, th
            y, x, th = precision_recall_curve(binary_label, prediction_score)

        # Calculate Calibration AUC
        calibration_auc = auc(x, y)

        return calibration_auc


class CalibrationCurveAUC(object):
    """Reliability Diagram for a binary prediction task, where
    binary label is the predictive correctness (or incorrectness)
    and the prediction score is the confidence score (or 1-confidence score).
  """
    def __init__(self, correct_pred_as_pos_label=False, n_bins=5, strategy='uniform'):
        self.correct_pred_as_pos_label = correct_pred_as_pos_label
        self.n_bins = n_bins
        self.strategy = strategy

    def auc(self, prob, target):
        # binary_label is the predicted correctness of the prediction
        selected = np.argmax(prob, axis=1)
        binary_label = target == selected
        prediction_score = np.max(prob, axis=1)

        if not self.correct_pred_as_pos_label:
            # Use incorrect prediction as the positive class.
            # This is important since an accurate model has few incorrect predictions.
            # This results in label imbalance in the calibration AUC computation, and
            # can lead to overly optimistic results.
            prediction_score = 1. - prediction_score
            binary_label = 1. - binary_label

        prob_true, prob_pred = calibration_curve(binary_label, prediction_score,
                                                 n_bins=self.n_bins,
                                                 strategy=self.strategy)

        # Calculate Calibration curve AUC
        calibration_auc = auc(prob_pred, prob_true)

        return calibration_auc


class AUCROCscore(object):
    """AUC-ROC scorere
    multi_class = 'ovr' one-vs-rest. sensitive to class imbalance
    multi_class = 'ovo' one-vs-one. insensitive to class imbalanced
  """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def score(self, prob, target):
        # binary_label is the predicted correctness of the prediction
        # insensity to class imba
        score = roc_auc_score(target, prob, multi_class='ovo')
        return score

def quadratic_uncertainty(probs, as_vec=False):
    # probs = samples, classes
    if as_vec:
        return 1 - (probs**2).sum(axis=-1)
    else:
        return np.mean(1 - (probs**2).sum(axis=-1))
