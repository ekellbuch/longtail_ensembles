from typing import Any

import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
from torchmetrics import F1Score as F1
from .schduler import WarmupCosineLR, CosineAnnealingLRWarmup
from .loss.BalancedSoftmaxLoss import BalancedSoftmax
from .loss.WeightedSoftmaxLoss import WeightedSoftmax
from .loss.WeightedCrossEntropyLoss import WeightedCrossEntropy
from .loss.FocalLoss import FocalLoss
from .loss.ECELoss import _ECELoss
from .models import all_classifiers


class CIFAR10_Models(pl.LightningModule):
    """Abstract base class for CIFAR10 Models
  """

    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)

    def forward(x):
        raise NotImplementedError

    def training_step(self):
        raise NotImplementedError

    def calibration(self):
        """Calculates binned calibration metrics given
        """
        raise NotImplementedError

    def forward_outs(self):
        raise NotImplementedError

    def validation_step(self, batch, batch_nb):
        outputs = self.forward(batch)
        self.log("loss/val", outputs['loss'])
        self.log("acc/val", outputs['accuracy'])

    def test_step(self, batch, batch_nb):
        outputs = self.forward(batch)
        self.log("acc/test", outputs['accuracy'])

        if 'ece' in outputs.keys():
            self.log("ece/test", outputs['ece'])

        if 'accuracy/per_model/model_0' in outputs.keys():
            for i in range(self.nb_models):
                self.log(f"acc/per_model/model_{i}", outputs[f'accuracy/per_model/model_{i}'])

    def setup_scheduler(self, optimizer, total_steps, warm_steps=None):
        """Chooses between the cosine learning rate scheduler that came with the repo, or step scheduler based on wideresnet training.

    """
        if self.hparams.scheduler in [None, "cosine"]:
            scheduler = {
                "scheduler":
                WarmupCosineLR(
                    optimizer,
                    warmup_epochs=total_steps * 0.3,
                    max_epochs=total_steps,
                    warmup_start_lr=1e-8,
                    eta_min=1e-8,
                ),
                "interval":
                "step",
                "name":
                "learning_rate",
            }
        elif self.hparams.scheduler == "cosine_fwarmup":
            scheduler = {
                "scheduler":
                WarmupCosineLR(
                    optimizer,
                    warmup_epochs=warm_steps,
                    max_epochs=total_steps,
                    warmup_start_lr=1e-8,
                    eta_min=1e-8,
                ),
                "interval":
                "step",
                "name":
                "learning_rate",
            }
        elif self.hparams.scheduler == "cosine_anneal":
            scheduler = {
                "scheduler":
                    CosineAnnealingLRWarmup(
                optimizer=optimizer,
                T_max=total_steps,
                eta_min=0.0,
                warmup_epochs=warm_steps,
                base_lr=0.05,
                warmup_lr=0.1
            ),
                "interval":
                    "step",
                "name":
                    "learning_rate",
            }

        elif self.hparams.scheduler == "single_step":
            scheduler = {
                "scheduler":
                    torch.optim.lr_scheduler.StepLR(optimizer,
                                                    gamma=0.1,
                                                    step_size=3
                                                    ),
                "interval":
                    "epoch",
                "frequency":
                    1,
                "name":
                    "learning_rate",
            }
        elif self.hparams.scheduler == "step":
            scheduler = {
                "scheduler":
                torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[60, 120, 160],
                                                     gamma=0.2,
                                                     last_epoch=-1),
                "interval":
                "epoch",
                "frequency":
                1,
                "name":
                "learning_rate",
            }
        elif self.hparams.scheduler == "step_ldam":
            # https://github.com/kaidic/LDAM-DRW/blob/2536330f2afdaa65618323cb5a5850efccce762a/cifar_train.py#L366
            scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                            start_factor=1/5,
                                                            end_factor=1,
                                                            total_iters=5)
            scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[160, 180],
                                                            gamma=0.1
                                                            )
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ChainedScheduler(
                    [scheduler1, scheduler2]),
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
            }
        elif self.hparams.scheduler == "step_ldam_v2":
            # https://github.com/kaidic/LDAM-DRW/blob/2536330f2afdaa65618323cb5a5850efccce762a/cifar_train.py#L366
            scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                            start_factor=1/5,
                                                            end_factor=1,
                                                            total_iters=5)
            scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[60, 120, 160, 180],
                                                            gamma=0.2,
                                                            last_epoch=-1,
                                                            )
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ChainedScheduler(
                    [scheduler1, scheduler2]),
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
            }
        elif self.hparams.scheduler == "cosine_balms":
            scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                            start_factor=1/2,
                                                            end_factor=1,
                                                            total_iters=warm_steps)
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=total_steps,
                                                            eta_min=0.0,
                                                                    )
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ChainedScheduler(
                    [scheduler1, scheduler2]),
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            }
        elif self.hparams.scheduler == "lambdalr":
            scheduler = {
                "scheduler":
                torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda epoch: 0.1**(epoch // 30)),
                "interval":
                "epoch",
                "frequency":
                1,
                "name":
                "learning_rate",
            }

        return scheduler


class CIFAR10Module(CIFAR10_Models):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.label_smoothing = self.hparams.get('label_smoothing', 0.0)
        self.criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing)
        self.num_classes = hparams.get('num_classes', 10)
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.ece = _ECELoss()

        self.model = all_classifiers[self.hparams.classifier](
            num_classes=self.num_classes)

        self.track_score_pclass = hparams.get('track_score_pclass', False)

        # additional metrics
        self.f1 = F1(num_classes=self.num_classes, average=None, task="multiclass")

    def predict_step(self, batch,batch_idx, dataloader_idx=0):
        images, labels = batch
        predictions = self.model(images)
        return predictions, labels

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        ece = self.ece(predictions, labels)
        outputs = {'loss': loss, 'accuracy': accuracy, 'ece': ece}

        # additional metrics
        f1 = self.f1(predictions, labels)

        outputs['f1'] = f1
        return outputs

    def forward_outs(self, batch, use_softmax=True, store_split=False):
        """Like forward, but just exit with the softmax predictions and labels. .
    """
        del store_split
        softmax = torch.nn.Softmax(dim=1)
        images, labels = batch
        predictions = self.model(images)
        if use_softmax:
            smpredictions = softmax(predictions)
        else:
            smpredictions = predictions
        return smpredictions, labels

    def training_step(self, batch, batch_nb):
        outputs = self.forward(batch)
        loss = outputs['loss']

        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        self.log("loss/train", loss)
        self.log("acc/train", outputs['accuracy'])

        # add f1 metrics per class
        if self.track_score_pclass:
            for i in range(self.num_classes):
                self.log(f"f1/per_class/class_{i}", outputs['f1'][i])

        return loss

    def validation_step(self, batch, batch_nb):
        outputs = self.forward(batch)
        self.log("loss/val", outputs['loss'])
        self.log("acc/val", outputs['accuracy'])

        # add f1 metrics per class
        if self.track_score_pclass:
            for i in range(self.num_classes):
                self.log(f"f1/per_class/class_{i}", outputs['f1'][i])

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
            momentum=0.9,
            nesterov=True,
        )
        self.trainer.fit_loop.setup_data()
        total_steps = self.trainer.max_epochs * len(self.trainer.train_dataloader)
        if self.hparams.get("warmup_epochs", None):
            warm_steps = self.hparams.warmup_epochs * len(self.trainer.train_dataloader)
        else:
            warm_steps = 0
        """
        # divide total step and warm step by gpus, not working
        if len(self.trainer.device_ids) > 1:
            num_gpus = len(self.trainer.device_ids)
            total_steps = total_steps // num_gpus
            warm_steps = warm_steps // num_gpus
        """
        scheduler = self.setup_scheduler(optimizer, total_steps, warm_steps)
        return [optimizer], [scheduler]


class CIFAR10EnsembleModule(CIFAR10_Models):
    """Customized module to train an ensemble of models independently
  """
    def __init__(self, hparams):
        super().__init__(hparams)
        self.nb_models = hparams.nb_models
        self.label_smoothing = self.hparams.get('label_smoothing', 0.0)
        self.criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing)
        self.fwd_criterion = torch.nn.NLLLoss()
        self.num_classes = hparams.get('num_classes', 10)
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.models = torch.nn.ModuleList([
            all_classifiers[self.hparams.classifier](
                num_classes=self.num_classes) for i in range(self.nb_models)
        ])  # now we add several different instances of the model.
        self.track_score_pmodel = hparams.get('track_score_pmodel', True)
        self.track_score_pclass = hparams.get('track_score_pclass', False)

        # additional metrics
        #self.f1 = F1(num_classes=self.num_classes, average=None, task="multiclass")
        #self.ece = _ECELoss()

    def forward(self, batch):
        """for forward pass, we want to take the softmax,
    aggregate the ensemble output, take log(\bar{f}) and apply NNLoss.
    prediction  = \bar{f}
    """
        images, labels = batch
        softmax = torch.nn.Softmax(dim=1)
        outputs= {}
        softmaxes = []
        for m_idx, m in enumerate(self.models):
            predictions = m(images)

            normed = softmax(predictions)
            softmaxes.append(normed)

            if self.track_score_pmodel:
                mloss = self.criterion(predictions, labels)
                accuracy = self.accuracy(predictions, labels)
                outputs[f'loss/per_model/model_{m_idx}'] = mloss
                outputs[f'accuracy/per_model/model_{m_idx}'] = accuracy

        # take average of all the ensemble outputs
        mean = torch.mean(torch.stack(softmaxes), dim=0)
        # we can pass this  through directly to the accuracy function.
        logoutput = torch.log(mean)
        tloss = self.fwd_criterion(
            logoutput, labels
        )  #  beware: this is a transformed input, don't evaluate on test loss of ensembles.
        accuracy = self.accuracy(mean, labels)

        outputs['loss'] = tloss
        outputs['accuracy'] = accuracy

        return outputs

    def forward_outs(self, batch, use_softmax=True, store_split=False):
        """Like forward, but just exit with the predictions and labels. .
    """
        images, labels = batch
        softmax = torch.nn.Softmax(dim=1)

        softmaxes = []
        for m in self.models:
            predictions = m(images)
            if use_softmax:
                predictions = softmax(predictions)
            softmaxes.append(predictions)
        if store_split:
            mean = torch.stack(softmaxes, 1)
        else:
            mean = torch.mean(torch.stack(softmaxes), dim=0)
        return mean, labels

    def training_step(self, batch, batch_nb):
        images, labels = batch
        losses = []
        accs = []

        outputs = {}
        for m_idx, m in enumerate(self.models):
            predictions = m(images)
            mloss = self.criterion(predictions, labels)
            accuracy = self.accuracy(predictions, labels)
            losses.append(mloss)
            accs.append(accuracy)
            if self.track_score_pmodel:
                outputs[f'loss/per_model/model_{m_idx}'] = mloss
                outputs[f'accuracy/per_model/model_{m_idx}'] = accuracy

        loss = sum(losses) / self.nb_models
        avg_accuracy = sum(accs) / self.nb_models

        outputs['loss'] = loss
        outputs['accuracy'] = avg_accuracy

        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        self.log("loss/train", loss)
        self.log("acc/train", outputs['accuracy'])

        if self.track_score_pmodel:
            for i in range(self.nb_models):
                self.log(f"acc/per_model/model_{i}", outputs[f'accuracy/per_model/model_{i}'])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.models.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
            momentum=0.9,
            nesterov=True,
        )
        self.trainer.fit_loop.setup_data()
        total_steps = self.trainer.max_epochs * len(self.trainer.train_dataloader)
        scheduler = self.setup_scheduler(optimizer, total_steps)
        return [optimizer], [scheduler]


class CIFAR10EnsembleJGAPModule(CIFAR10EnsembleModule):
    """
  Formulation of the ensemble as a regularized single model with variable weight on regularization.

  """

    def __init__(self, hparams):
        super().__init__(hparams)
        self.traincriterion = NLLLoss_label_smooth(self.num_classes,
                                                   self.label_smoothing)
        self.gamma = hparams.gamma

    def training_step(self, batch, batch_nb):
        """When we train, we want to train independently.
    Loss = NLL(log \bar{f}, y ) + gamma*JGAP(softmaxes, label)
    JGAP = 1/M sum_i^M CE(f_i,y) - NLL(log \bar{f}, y )
    """
        softmax = torch.nn.Softmax(dim=1)
        outputs = {}
        images, labels = batch
        losses, accs, eces, f1s = [], [], [], []
        softmaxes = []

        for m_idx, m in enumerate(self.models):
            predictions = m(images)
            mloss = self.criterion(predictions, labels)
            accuracy = self.accuracy(predictions, labels)
            normed = softmax(predictions)
            softmaxes.append(normed)
            losses.append(mloss)
            accs.append(accuracy)
            if self.track_score_pmodel:
                outputs[f'loss/per_model/model_{m_idx}'] = mloss
                outputs[f'accuracy/per_model/model_{m_idx}'] = accuracy

        logoutput = torch.log(torch.mean(torch.stack(softmaxes), dim=0))
        mloss = self.traincriterion(logoutput, labels)

        # jensen gap
        avg_sm_loss = sum(losses) / self.nb_models
        jgaploss = avg_sm_loss - mloss

        loss = (mloss + self.gamma * jgaploss)
        accuracy = self.accuracy(logoutput, labels)

        outputs['loss'] = loss
        outputs['accuracy'] = accuracy

        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        self.log("loss/train", loss)
        self.log("acc/train", outputs['accuracy'])

        if self.track_score_pmodel:
            for i in range(self.nb_models):
                self.log(f"acc/per_model/model_{i}", outputs[f'accuracy/per_model/model_{i}'])

        return loss


class CIFAR10BLossModule(CIFAR10Module):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.label_smoothing = self.hparams.get('label_smoothing', 0.0)
        if self.label_smoothing > 0:
            raise NotImplementedError(
                "Label smoothing not implemented for BalancedLoss")
        self.criterion = BalancedSoftmax(
            torch.tensor(self.hparams.samples_per_class))


class CIFAR10WeightedSoftmaxLossModule(CIFAR10Module):
    """
    Weighted softmax loss L = 1/pi_y log (p_y)  for class y
    where pi_y = n_y / n, and n_y is the sample number of class y.
    """
    def __init__(self, hparams):
        super().__init__(hparams)
        self.label_smoothing = self.hparams.get('label_smoothing', 0.0)
        self.criterion = WeightedSoftmax(
            torch.tensor(self.hparams.samples_per_class), label_smoothing=self.label_smoothing)


class CIFAR10FocalLossModule(CIFAR10Module):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.label_smoothing = self.hparams.get('label_smoothing', 0.0)
        self.focal_gamma = self.hparams.get('focal_gamma', 0.0)
        self.focal_size_average = self.hparams.get('focal_size_average', False)
        if self.label_smoothing > 0:
            raise NotImplementedError(
                "Label smoothing not implemented for FocalLoss")
        self.criterion = FocalLoss(
            gamma=self.focal_gamma, size_average=self.focal_size_average)

class CIFAR10WeightedCrossEntropyLossModule(CIFAR10Module):
    """
    Weighted softmax loss L = 1/pi_y log (p_y)  for class y
    where pi_y = n_y / n, and n_y is the sample number of class y.
    """
    def __init__(self, hparams):
        super().__init__(hparams)
        self.label_smoothing = self.hparams.get('label_smoothing', 0.0)
        self.criterion = WeightedCrossEntropy(
            torch.tensor(self.hparams.samples_per_class), label_smoothing=self.label_smoothing)


class CIFAR10EnsembleJGAPBLossModule(CIFAR10EnsembleJGAPModule):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.label_smoothing = self.hparams.get('label_smoothing', 0.0)
        if self.label_smoothing != 0:
            raise NotImplementedError(
                "Label smoothing not implemented for BalancedLoss")
        self.criterion = BalancedSoftmax(
            torch.tensor(self.hparams.samples_per_class))
        if self.gamma != 1:
            raise NotImplementedError("Gamma not implemented for BalancedLoss")

class NLLLoss_label_smooth(torch.nn.Module):

    def __init__(self, num_classes, label_smoothing=0.1):
        super(NLLLoss_label_smooth, self).__init__()
        self.negative = label_smoothing / (num_classes - 1)
        self.positive = (1 - label_smoothing)

    def forward(self, log_softmax, target):
        true_dist = torch.zeros_like(log_softmax)
        true_dist.fill_(self.negative)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.positive)
        return torch.sum(-true_dist * log_softmax, dim=1).mean()


all_modules = {
    "base": CIFAR10Module,
    "base_bloss": CIFAR10BLossModule,
    "focal_loss": CIFAR10FocalLossModule,
    "weighted_softmax": CIFAR10WeightedSoftmaxLossModule,
    "weighted_ce": CIFAR10WeightedCrossEntropyLossModule,
    "ensemble": CIFAR10EnsembleModule,  # train time ensemble
    "ensemble_jgap": CIFAR10EnsembleJGAPModule,  # train time ensemble
    "ensemble_jgap_bloss": CIFAR10EnsembleJGAPBLossModule,  # train time ensemble
}
