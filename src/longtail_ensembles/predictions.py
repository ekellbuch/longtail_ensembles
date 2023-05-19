"""
Classes to collect prediction from a single model or an ensemble.
"""
import os
import h5py
import numpy as np
import pandas as pd
from .metrics import AccuracyData, NLLData, BrierScoreData, \
    F1ScoreData, PrecisionScoreData, RecallScoreData, \
    CalibrationData, quadratic_uncertainty, CalibrationAUC, VarianceData
from .utils import add_dicts
import itertools
from .diversity import diversity_metrics

class Model(object):

    def __init__(self, modelprefix, data):
        self.modelprefix = modelprefix
        self.data = data

        self._logits = None
        self._labels = None
        self._probs = None

    def register(self,
                 filename,
                 inputtype=None,
                 labelpath=None,
                 logits=True,
                 npz_flag=None,
                 mask_array=None):
        """Register a model's predictions to this model object.
    :param filename: (string) path to file containing logit model predictions.
    :param modelname: (string) name of the model to identify within an ensemble
    :param inputtype: (optional) [h5,npy] h5 inputs or npy input, depending on if we're looking at imagenet or cifar10 datasets. If not given, will be inferred from filename extension.
    :param labelpath: (optional) if npy format files, labels must be given.
    :param logits: (optional) we assume logits given, but probs can also be given directly.
    :param npz_flag: if filetype is .npz, we asume that we need to pass a dictionary key to retrieve logits
          used for `cifar10` or `cinic10` logits.
    """

        if mask_array is None:
            mask_array = ()
        self.filename = filename
        self.labelpath = labelpath

        if inputtype is None:
            _, ext = os.path.splitext(filename)
            inputtype = ext[1:]
            assert inputtype in [
                "h5", "hdf5", "npy", "npz", "pickle"
            ], "inputtype inferred from extension must be `h5` or `npy`, or `npz` if not given, not {}.".format(
                inputtype)

        if mask_array is None:
            mask_array = ()

        if inputtype in ["h5", "hdf5"]:
            with h5py.File(str(filename), 'r') as f:
                self._logits = f['logits'][mask_array]
                self._labels = f['targets'][mask_array].astype('int')
                self._probs = np.exp(self._logits) / np.sum(
                    np.exp(self._logits), 1, keepdims=True)

        elif inputtype == "npy":
            assert labelpath is not None, "if npy, must give labels."
            if logits:
                self._logits = np.load(filename)[mask_array]
                self._labels = np.load(labelpath)[mask_array].astype('int')
                self._probs = np.exp(self._logits) / np.sum(
                    np.exp(self._logits), 1, keepdims=True)
            else:
                self._logits = None
                self._labels = np.load(labelpath)
                self._probs = np.load(filename)

        elif inputtype == "npz":
            assert labelpath is not None, "if npz must give labels."
            assert npz_flag is not None, "if npz must give flag for which logits to retrieve."
            if logits:
                self._logits = np.load(filename)[npz_flag][mask_array]
                self._labels = np.load(labelpath)[mask_array].astype('int')
                self._probs = np.exp(self._logits) / np.sum(
                    np.exp(self._logits), 1, keepdims=True)
            else:
                self._logits = None
                self._labels = np.load(labelpath)[mask_array]
                self._probs = np.load(filename)[mask_array]
        elif inputtype == 'pickle':
            if logits:
                self._logits = pd.read_pickle(filename)['logits']
                self._labels = np.load(labelpath).astype('int')
                self._probs = np.exp(self._logits) / np.sum(np.exp(self._logits), 1, keepdims=True)

    def masking(self, mask_array):
        self._logits = self._logits[mask_array]
        self._labels = self._labels[mask_array]
        self._probs = self._probs[mask_array]

    #@property
    def probs(self):
        # n x c
        return self._probs

    #@property
    def labels(self):
        # n
        return self._labels

    #@property
    def logits(self):
        # n x c
        return self._logits

    def get_accuracy(self):
        ad = AccuracyData()
        return ad.accuracy(self.probs(), self.labels())

    def get_accuracy_per_class(self, **kwargs):
        ad = AccuracyData()
        return ad.accuracy_per_class(self.probs(), self.labels(), **kwargs)

    def get_nll(self, normalize=True):
        nld = NLLData()
        return nld.nll(self.probs(), self.labels(), normalize=normalize)

    def get_nll_vec(self):
        nld = NLLData()
        return nld.nll_vec(self.probs(), self.labels())

    def get_brier(self):
        bsd = BrierScoreData()
        return bsd.brierscore_multi(self.probs(), self.labels())

    def get_true_probs(self):
        # get the probability for the correct class
        return self.probs()[np.arange(self.probs().shape[0]), self.labels()]

    def get_f1score(self, average='macro'):
        # get the probability for the correct class
        f1score = F1ScoreData()
        return f1score.f1_score(self.probs(), self.labels(), average=average)

    def get_f1score_per_class(self, **kwargs):
        # get the probability for the correct class
        f1score = F1ScoreData()
        return f1score.f1_score_per_class(self.probs(), self.labels(), **kwargs)

    def get_precision(self, average='macro'):
        # get the probability for the correct class
        score = PrecisionScoreData()
        return score.precision_score(self.probs(),
                                     self.labels(),
                                     average=average)

    def get_recall(self, average='macro'):
        # get the probability for the correct class
        score = RecallScoreData()
        return score.recall_score(self.probs(), self.labels(), average=average)

    def get_ece(self, power=1, num_bins=15):
        ece = CalibrationData(power=power, num_bins=num_bins)
        return ece.ece(self.probs(), self.labels())

    def get_qunc(self, as_vec=False):
        """estimate the average single model uncertainty.

        """
        all_probs = self.probs()  # (samples,classes)
        return quadratic_uncertainty(all_probs, as_vec=as_vec)

    def get_calibration_roc_auc(self, correct_pred_as_pos_label=False):
        """
        calculate calibrationAUC score
        """
        score = CalibrationAUC(curve='roc', correct_pred_as_pos_label=correct_pred_as_pos_label)
        return score.auc(self.probs(), self.labels())

    def get_calibration_pr_auc(self, correct_pred_as_pos_label=True):
        """
        calculate calibrationAUC score
        """
        score = CalibrationAUC(curve='pr', correct_pred_as_pos_label=correct_pred_as_pos_label)
        return score.auc(self.probs(), self.labels())

    def get_class_variance(self, as_vec=False):
        """
        Calculate the variance across the classes for each sample.
        :param as_vec:
        :return:
        """
        var_ = np.var(self.probs(), axis=-1) # samples, classes
        if as_vec:
            return var_
        return np.mean(var_)

    def get_class_variance_per_class(self, as_vec=False):
        """
        Calculate the spread in the predictions, separate given true targets
        :param as_vec:
        :return:
        """
        num_classes = self.probs().shape[-1]
        var_ = np.var(self.probs(), axis=-1)  # samples, classes
        all_class_keys = np.arange(num_classes)
        per_class_var = {}
        for class_label in all_class_keys:
            mask_ = np.argwhere(self.labels() == class_label)
            per_class_var[class_label] = var_[mask_]

            if not as_vec:
                # average across samples
                per_class_var[class_label] = np.mean(per_class_var[class_label])

        return per_class_var

    def get_variance(self, as_vec=False):
        return np.nan

class EnsembleModel(Model):
    """Collect the outputs of a series of models to allow ensemble based analysis.

  :param modelprefix: string prefix to identify this set of models.
  :param data: string to identify the dataset that set of models is evaluated on.
  """

    def __init__(self, modelprefix, data):
        super().__init__(modelprefix, data)
        self.models = {
        }  ## dict of dicts- key is modelname, value is dictionary of preds/labels.

    def register(self,
                 filename,
                 modelname,
                 inputtype=None,
                 labelpath=None,
                 logits=True,
                 npz_flag=None,
                 mask_array=None):
        """Register a model's predictions to this ensemble object.
    :param filename: (string) path to file containing logit model predictions.
    :param modelname: (string) name of the model to identify within an ensemble
    :param inputtype: (optional) [h5,npy] h5 inputs or npy input, depending on if we're looking at imagenet or cifar10 datasets. If not given, will be inferred from filename extension.
    :param labelpath: (optional) if npy format files, labels must be given.
    :param logits: (optional) we assume logits given, but probs can also be given directly.
    :param npz_flag: if filetype is .npz, we asume that we need to pass a dictionary key to retrieve logits
          used for `cifar10` or `cinic10` logits.
    """
        model = Model(modelname, 'data')
        model.register(filename=filename,
                       inputtype=inputtype,
                       labelpath=labelpath,
                       logits=logits,
                       npz_flag=npz_flag,
                       mask_array=mask_array)
        self.models[modelname] = model

    def probs(self):
        """Calculates mean confidence across all softmax output.

    :return: array of shape samples, classes giving per class variance.
    """
        all_probs = []
        for model, modeldata in self.models.items():
            probs = modeldata.probs()
            all_probs.append(probs)
        array_probs = np.stack(all_probs, axis=0)
        self._probs = np.mean(array_probs, axis=0)
        return self._probs

    def labels(self):
        for model, modeldata in self.models.items():
            self._labels = modeldata.labels()
            break
        return self._labels

    def logits(self):
        raise NotImplementedError

    def get_variance(self, as_vec=False):
        """Get variance across ensemble members. Estimate sample variance across the dataset with unbiased estimate.
        not using target information.

        The variances across the ensemble predictions is summed across classes, and then averaged over samples.
        measures the dispersion of the ensemble predictions.
    """
        all_probs = []
        for model, modeldata in self.models.items():
            probs = modeldata.probs()
            all_probs.append(probs)
        array_probs = np.stack(all_probs, axis=0)  # (models,samples,classes)
        var = np.var(array_probs, axis=0,
                     ddof=1)  # variance across models (samples, classes)

        var = np.sum(var, axis=-1)  # sum over classes (samples)
        if as_vec:
            return var

        return np.mean(var)  # average over samples

    def get_bias_bs(self, as_vec=False):
        """Given a brier score, estimate bias across the dataset.
        using target information
    """
        bsd = BrierScoreData()
        model_bs = np.array([
            bsd.brierscore_multi_vec(m.probs(), m.labels())
            for m in self.models.values()
        ])

        if as_vec:
            return np.mean(model_bs, axis=0)
        return np.mean(np.mean(model_bs, axis=0))

    def get_avg_nll(self, as_vec=False):
        """estimate the average NLL across the ensemble.
        using target information
    """
        all_nll = []
        for model, modeldata in self.models.items():
            probs = modeldata.probs()
            targets = modeldata.labels()
            all_nll.append(-np.log(probs[np.arange(len(targets)), targets]))

        array_nll = np.stack(all_nll, axis=0)  # (models,samples)
        if as_vec:
            return np.mean(array_nll, axis=0)  # take mean across models
        return np.mean(np.mean(array_nll, axis=0))

    def get_nll_div(self, as_vec=False):
        """estimate diversity between ensembles members corresponding to the jensen gap between ensemble and single
    model nll
        using target information

    """
        all_probs = []
        for model, modeldata in self.models.items():
            probs = modeldata.probs()
            targets = modeldata.labels()
            all_probs.append(probs[np.arange(len(targets)), targets])

        array_probs = np.stack(all_probs, axis=0)  # (models,samples)
        norm_term = np.log(np.mean(array_probs, axis=0))
        diversity = -np.mean(np.log(array_probs), axis=0) + norm_term
        if as_vec:
            return diversity
        return np.mean(diversity)

    def get_pairwise_corr(self):
        """Get pairwise correlation between ensemble members:
        Does not use target information

    """
        all_probs = []
        M = len(self.models)
        for model, modeldata in self.models.items():
            probs = modeldata.probs()
            all_probs.append(probs)
        array_probs = np.stack(all_probs, axis=0)  # (models,samples,classes)
        array_pred_labels = np.argmax(array_probs, axis=-1)  # (models,samples)
        compare = array_pred_labels[:, None, :] == array_pred_labels  # (models, models,samples)
        means = np.mean(compare.astype(int), axis=-1)  # (models,models)
        mean_means = np.sum(np.tril(means, k=-1)) / (
            (M * (M - 1)) / 2)  # (average across all pairs)
        return 1 - mean_means

    def get_variance_true_probs(self, as_vec=False):
        """Get variance across ensemble members for the correct prediction
        var(f_i=y) where f_i is the prob for the prediction of model
        and y is the true label.
        """
        all_probs = []
        for model, modeldata in self.models.items():
            probs = modeldata.get_true_probs()  # samples
            all_probs.append(probs)
        array_probs = np.stack(all_probs, axis=0)  # (models,samples)
        var = np.var(array_probs, axis=0, ddof=1)  # (samples)
        if as_vec:
            return var
        return np.mean(var)  # average over samples

    def get_variance_predictive(self, as_vec=False):
        """Get predicted variance across ensemble members.
        for the predicted class
        var(f_i) where f_i is the prob for the prediction of model i
        """
        # Find the predicted class
        all_probs = []
        for model, modeldata in self.models.items():
            probs = modeldata.probs()  # (samples, classes)
            all_probs.append(probs)
        array_probs = np.stack(all_probs, axis=0)  # (models,samples, classes)
        # get the predicted class
        ensemble_probs = np.mean(array_probs, axis=0)  # (samples, classes)
        ensemble_pred = np.argmax(ensemble_probs, axis=-1)  # (samples)
        # get the variance for the predicted class
        array_probs = array_probs[:,
                                  np.arange(len(ensemble_pred)),
                                  ensemble_pred]  # (models, samples)
        var = np.var(array_probs, axis=0, ddof=1)  # (samples)
        if as_vec:
            return var
        return np.mean(var)  #average over samples

    def get_variance_per_class(self, as_vec=False):
        """Get prob variance across ensemble members for each true class:

        The variances across the ensemble predictions is averaged over samples if as_vec=False
    """
        all_probs = []
        all_labels = []
        for model, modeldata in self.models.items():
            probs = modeldata.probs()
            labels = modeldata.labels()
            all_probs.append(probs)
            all_labels.append(labels)
        array_probs = np.stack(all_probs, axis=0)  # (models,samples,classes)
        array_labels = np.stack(all_labels, axis=0)  # (models,samples)
        # check all models have the same labels
        assert np.all(array_labels == array_labels.mean(axis=0))
        array_labels = array_labels[0]  # (samples)
        var = np.var(array_probs, axis=0,
                     ddof=1)  # variance across models (samples, classes)

        num_samples, num_classes = var.shape
        all_class_keys = np.arange(num_classes)
        per_class_var = {}
        for class_label in all_class_keys:
            # not the variance across the samples for the true class
            # but the sum variance over the classes when the true class in class_label
            mask_ = np.argwhere(array_labels == class_label)
            per_class_var[class_label] = np.sum(var[mask_], axis=1)  # samples in class l_

            if not as_vec:
                # average the variance across samples
                per_class_var[class_label] = np.mean(per_class_var[class_label])

        return per_class_var

    def get_cv_per_class(self, as_vec=False):
        """Get coefficient of variation across ensemble members for each true class:

        The variances across the ensemble predictions is averaged over samples if as_vec=False
    """
        all_probs = []
        all_labels = []
        for model, modeldata in self.models.items():
            probs = modeldata.probs()
            labels = modeldata.labels()
            all_probs.append(probs)
            all_labels.append(labels)
        array_probs = np.stack(all_probs, axis=0)  # (models,samples,classes)
        array_labels = np.stack(all_labels, axis=0)  # (models,samples)
        # check all models have the same labels
        assert np.all(array_labels == array_labels.mean(axis=0))
        array_labels = array_labels[0]  # (samples)
        var = np.std(array_probs, axis=0, ddof=1)  # std across models (samples, classes)
        mean_ = np.mean(array_probs, axis=0)  # mean across models (samples, classes)
        cv_ = var/mean_ # coefficient of variation across models
        num_samples, num_classes = var.shape
        all_class_keys = np.arange(num_classes)
        per_class_var = {}
        for class_label in all_class_keys:
            # sum of the coefficients of variation
            mask_ = np.argwhere(array_labels == class_label)
            per_class_var[class_label] = np.sum(cv_[mask_], axis=1)  # samples in class l_
            if not as_vec:
                # average the variance across samples
                per_class_var[class_label] = np.mean(per_class_var[class_label])

        return per_class_var

    def _get_diversity_score(self, metric):
        """Get average disagreement between ensemble members:

        """
        all_probs = []
        all_disagreements = []
        M = len(self.models)
        for model, modeldata in self.models.items():
            probs = modeldata.probs()
            all_probs.append(probs)
        array_probs = np.stack(all_probs, axis = 0) # (models,samples,classes)
        num_samples = array_probs.shape[1]
        for pair in list(itertools.combinations(range(M), 2)):
            p_ = array_probs[pair[0]]
            q_ = array_probs[pair[1]]
            disagr_ = np.sum(diversity_metrics[metric](p_, q_))
            all_disagreements.append(disagr_)
        return np.mean(all_disagreements)/num_samples

    def _get_diversity_score_per_class(self, metric):
        """Get average disagreement between ensemble members:

        """
        all_probs = []
        for model, modeldata in self.models.items():
            probs = modeldata.probs()
            targets = modeldata.labels()
            all_probs.append(probs)
        array_probs = np.stack(all_probs, axis = 0) # (models,samples,classes)
        M, num_samples, num_classes = array_probs.shape

        all_disagreements = {}

        for class_label in range(num_classes):
            cls_disagreements = []
            mask_ = np.argwhere(targets == class_label)
            for pair in list(itertools.combinations(range(M), 2)):
                p_ = array_probs[pair[0]][mask_]
                q_ = array_probs[pair[1]][mask_]
                disagr_ = np.sum(diversity_metrics[metric](p_, q_))
                cls_disagreements.append(disagr_)

            all_disagreements[class_label] = np.mean(cls_disagreements)/len(mask_)

        return all_disagreements

    def get_diversity_score(self, metric='pairwise_corr'):

        if metric == 'pairwise_corr':
            return self.get_pairwise_corr()
        else:
            return self._get_diversity_score(metric=metric)

    def get_diversity_score_per_class(self, metric='pairwise_corr'):

        if metric == 'pairwise_corr':
            raise NotImplementedError
        else:
            return self._get_diversity_score_per_class(metric=metric)

    def _get_dkl_diffs(self, per_class=True, metric='dkl_qbar_qi'):
        """Get coefficient of variation across ensemble members for each true class:

        The variances across the ensemble predictions is averaged over samples if as_vec=False
    """
        all_probs = []
        all_labels = []
        for model, modeldata in self.models.items():
            probs = modeldata.logits()
            labels = modeldata.labels()
            all_probs.append(probs)
            all_labels.append(labels)

        all_probs = np.stack(all_probs, axis=0)  # (models,samples,classes)
        all_labels = all_labels[0]  # (samples)


        score = VarianceData()
        dkl_diff = {
            'dkl_qbar_qi': score.dkl_qbar_qi,
            'dkl_uni_eq': score.dkl_uni_eq,
            'dkl_qdag_qi': score.dkl_qdag_qi,
        }
        output = dkl_diff[metric](logits=all_probs, target=all_labels, per_class=per_class)

        return output

    def get_dkl_qs(self, per_class=True, metric='dkl_qbar_qi'):
        return self._get_dkl_diffs(per_class=per_class, metric=metric)

class EnsembleModelLogit(EnsembleModel):
    """EnsembleModel where the ensemble prediction is given
     by averaging the logits across ensemble members.

  :param modelprefix: string prefix to identify this set of models.
  :param data: string to identify the dataset that set of models is evaluated on.
  """

    def __init__(self, modelprefix, data):
        super().__init__(modelprefix, data)
        #self._logits = None

    def logits(self):
        """ Calculate avergge logits from ensemble members.

    :return: get average logits from ensemble members.
    """
        all_probs = []
        for model, modeldata in self.models.items():
            probs = modeldata.logits()
            all_probs.append(probs)
        array_probs = np.stack(all_probs, axis=0)
        # average logits
        self._logits = np.mean(array_probs, axis=0)
        return self._logits

    def probs(self):
        """Calculates mean confidence across all softmax output.
    by averaging the logits and then applying softmax.
    :return: array of shape samples, classes
    """
        self._probs = np.exp(self.logits()) / np.sum(
            np.exp(self.logits()), 1, keepdims=True)

        return self._probs

    def get_variance(self, as_vec=False):
        """Get variance across ensemble members. Estimate sample variance across the dataset with unbiased estimate.
        not using target information.

        The variances across the ensemble logits is summed across classes, and then averaged over samples.
        measures the dispersion of the ensemble predictions.
    """
        all_probs = []
        for model, modeldata in self.models.items():
            probs = modeldata.logits()
            all_probs.append(probs)
        array_probs = np.stack(all_probs, axis=0)  # (models,samples,classes)
        var = np.var(array_probs, axis=0,
                     ddof=1)  # variance across models (samples, classes)

        var = np.sum(var, axis=-1)  # sum over classes (samples)
        if as_vec:
            return var

        return np.mean(var)  # average over samples

    def get_variance_per_class(self, as_vec=False):
        """Get logit variance across ensemble members for each true class:

        get variance across models, then sum across classes, then average over samples if as_vec=False

    """
        all_probs = []
        all_labels = []
        for model, modeldata in self.models.items():
            probs = modeldata.logits()
            labels = modeldata.labels()
            all_probs.append(probs)
            all_labels.append(labels)
        array_probs = np.stack(all_probs, axis=0)  # (models,samples,classes)
        array_labels = np.stack(all_labels, axis=0)  # (models,samples)
        # check all models have the same labels
        assert np.all(array_labels == array_labels.mean(axis=0))
        array_labels = array_labels[0]  # (samples)
        var = np.var(array_probs, axis=0,
                     ddof=1)  # variance across models (samples, classes)

        num_samples, num_classes = var.shape
        all_class_keys = np.arange(num_classes)
        per_class_var = {}
        for class_label in all_class_keys:
            # not the variance across the samples for the true class
            # but the sum variance over the classes when the true class in class_label
            mask_ = np.argwhere(array_labels == class_label)
            per_class_var[class_label] = np.sum(var[mask_], axis=1)  # samples in class l_

            if not as_vec:
                # average the variance across samples
                per_class_var[class_label] = np.mean(per_class_var[class_label])

        return per_class_var

    def get_cv_per_class(self, as_vec=False):
        """Get coefficient of variation across ensemble members for each true class:

        The variances across the ensemble predictions is averaged over samples if as_vec=False
    """
        all_probs = []
        all_labels = []
        for model, modeldata in self.models.items():
            probs = modeldata.logits()
            labels = modeldata.labels()
            all_probs.append(probs)
            all_labels.append(labels)
        array_probs = np.stack(all_probs, axis=0)  # (models,samples,classes)
        array_labels = np.stack(all_labels, axis=0)  # (models,samples)
        # check all models have the same labels
        assert np.all(array_labels == array_labels.mean(axis=0))
        array_labels = array_labels[0]  # (samples)
        var = np.std(array_probs, axis=0, ddof=1)  # std across models (samples, classes)
        mean_ = np.mean(array_probs, axis=0)  # mean across models (samples, classes)
        cv_ = var/mean_ # coefficient of variation across models
        num_samples, num_classes = var.shape
        all_class_keys = np.arange(num_classes)
        per_class_var = {}
        for class_label in all_class_keys:
            # sum of the coefficients of variation
            mask_ = np.argwhere(array_labels == class_label)
            per_class_var[class_label] = np.sum(cv_[mask_], axis=1)  # samples in class l_
            if not as_vec:
                # average the variance across samples
                per_class_var[class_label] = np.mean(per_class_var[class_label])

        return per_class_var


class MultipleModel(EnsembleModel):
    """
    Multiple models where metrics are the average of the metrics of the models.
    """
    def __init__(self, modelprefix, data):
        super().__init__(modelprefix, data)

    def probs(self):
       raise NotImplementedError

    def labels(self):
        raise NotImplementedError

    def logits(self):
        raise NotImplementedError

    def get_accuracy(self):
        num_models = len(self.models)
        avg_val = 0
        for model_idx, (model, modeldata) in enumerate(self.models.items()):
            avg_val += modeldata.get_accuracy()
        return avg_val / num_models

    def get_accuracy_per_class(self, **kwargs):
        num_models = len(self.models)
        avg_val = {}
        for model_idx, (model, modeldata) in enumerate(self.models.items()):
            new_val = modeldata.get_accuracy_per_class(**kwargs)
            avg_val = add_dicts(avg_val, new_val)
        avg_val = {k: v / num_models for k, v in avg_val.items()}
        return avg_val

    def get_f1score_per_class(self, **kwargs):
        num_models = len(self.models)
        avg_val = {}
        for model_idx, (model, modeldata) in enumerate(self.models.items()):
            new_val = modeldata.get_f1score_per_class(**kwargs)
            avg_val = add_dicts(avg_val, new_val)
        avg_val = {k: v / num_models for k, v in avg_val.items()}
        return avg_val

    def get_nll(self, normalize=True):
        num_models = len(self.models)
        avg_val = 0
        for model_idx, (model, modeldata) in enumerate(self.models.items()):
            avg_val += modeldata.get_nll(normalize=normalize)
        return avg_val / num_models

    def get_nll_vec(self):
        raise NotImplementedError

    def get_brier(self):
        num_models = len(self.models)
        avg_val = 0
        for model_idx, (model, modeldata) in enumerate(self.models.items()):
            avg_val += modeldata.get_brier()
        return avg_val / num_models

    def get_true_probs(self):
        # get the probability for the correct class
        raise NotImplementedError

    def get_f1score(self, average='macro'):
        num_models = len(self.models)
        avg_val = 0
        for model_idx, (model, modeldata) in enumerate(self.models.items()):
            avg_val += modeldata.get_f1score(average=average)
        return avg_val / num_models
    def get_precision(self, average='macro'):
        num_models = len(self.models)
        avg_val = 0
        for model_idx, (model, modeldata) in enumerate(self.models.items()):
            avg_val += modeldata.get_precision(average=average)
        return avg_val / num_models

    def get_recall(self, average='macro'):
        num_models = len(self.models)
        avg_val = 0
        for model_idx, (model, modeldata) in enumerate(self.models.items()):
            avg_val += modeldata.get_recall(average=average)
        return avg_val / num_models

    def get_ece(self, power=1, num_bins=15):
        num_models = len(self.models)
        avg_val = 0
        for model_idx, (model, modeldata) in enumerate(self.models.items()):
            avg_val += modeldata.get_ece(power=power, num_bins=num_bins)
        return avg_val/num_models

    def get_qunc(self, as_vec=False):
        num_models = len(self.models)
        avg_val = 0
        for model_idx, (model, modeldata) in enumerate(self.models.items()):
            avg_val += modeldata.get_qunc(as_vec=as_vec)
        return avg_val/num_models

    def get_calibration_roc_auc(self, **kwargs):
        """
        calculate calibrationAUC score
        """
        num_models = len(self.models)
        avg_val = 0
        for model_idx, (model, modeldata) in enumerate(self.models.items()):
            avg_val += modeldata.get_calibration_roc_auc(**kwargs)
        return avg_val / num_models

    def get_calibration_pr_auc(self, **kwargs):
        """
        calculate calibrationAUC score
        """
        num_models = len(self.models)
        avg_val = 0
        for model_idx, (model, modeldata) in enumerate(self.models.items()):
            avg_val += modeldata.get_calibration_pr_auc(**kwargs)
        return avg_val / num_models

    def get_class_variance(self):
        num_models = len(self.models)
        avg_val = 0
        for model_idx, (model, modeldata) in enumerate(self.models.items()):
            avg_val += modeldata.get_class_variance()
        return avg_val / num_models

    def get_class_variance_per_class(self, **kwargs):
        num_models = len(self.models)
        avg_val = {}
        for model_idx, (model, modeldata) in enumerate(self.models.items()):
            new_val = modeldata.get_class_variance_per_class(**kwargs)
            avg_val = add_dicts(avg_val, new_val)
        avg_val = {k: v / num_models for k, v in avg_val.items()}
        return avg_val