"""
Apply temperature scaling to a trained model.
References:
 https://github.com/torrvision/focal_calibration/blob/main/temperature_scaling.py
 https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

"""
import torch
from torch import nn, optim
from .loss.ECELoss import _ECELoss
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchmetrics import F1Score as F1

class ModuleWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, temperature):
        super(ModuleWithTemperature, self).__init__()
        self.model = model
        self.temperature = temperature

    def forward(self, input):
        # use temperature parameter to define the logits.
        logits = self.model(input)
        return self.temperature_scale(logits)
    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        return logits / self.temperature

class TemperatureScalingTask(pl.LightningModule):
    def __init__(self, module,
                 init_temp=1.5,
                 use_train_loss=False,
                 opt_params=None,
                 **kwargs,
                 ):
        super().__init__()
        self.init_temperature = init_temp
        self.optimizer_params = opt_params
        self.temperature = nn.Parameter(torch.ones(1) * self.init_temperature)
        self.ece_criterion = _ECELoss()
        self.num_classes = module.num_classes
        self.model = module.model

        if use_train_loss:
            self.criterion = module.criterion
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        return logits / self.temperature

    def forward(self, batch):
        images, labels = batch
        self.model.eval()
        with torch.no_grad():
            logits = self.model(images)
        ts_logits = self.temperature_scale(logits)
        loss = self.criterion(ts_logits, labels)
        accuracy = self.accuracy(ts_logits, labels)
        ece = self.ece_criterion(ts_logits, labels)

        outputs = {'loss': loss,
                   'accuracy': accuracy,
                   'ece': ece}

        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        self.log("ece/train", ece)

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        return outputs['loss']

    def configure_optimizers(self):
        if self.optimizer_params is None:
            optimizer_params = {'lr': 0.01,
                                'max_iter': 50,
                                'line_search_fn': 'strong_wolfe'}
        else:
            optimizer_params = self.optimizer_params
        optimizer = optim.LBFGS([self.temperature], **optimizer_params)
        return optimizer

    """
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        optimizer.step(closure=optimizer_closure)
        # example from https://github.com/Lightning-AI/lightning/issues/2382
        # step:
        optimizer.step(closure=optimizer_closure)

        # do not clamp with LBFGS!
        # clamp parameter
        for param in self.parameters():
            param.data = param.data.clamp(min=0.1)
    """
class ProbEnsembleTemperatureScalingTask(pl.LightningModule):
    def __init__(self,
                 models,
                 init_temp=1.5,
                 num_classes=10,
                 use_train_loss=False,
                 opt_params=None,
                 criterion=None,
                 **kwargs,
                 ):
        super().__init__()
        self.init_temperature = init_temp
        self.optimizer_params = opt_params
        self.temperature = nn.Parameter(torch.ones(1) * self.init_temperature)
        self.ece_criterion = _ECELoss()
        self.num_classes = num_classes
        self.models = models

        if use_train_loss and criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        Eq (8) in https://arxiv.org/pdf/2007.08792.pdf
        """
        return torch.log(logits) / self.temperature

    def _call_to_log_outputs(self, outputs, prefix):
        self.log(f"loss/{prefix}", outputs['loss'])
        self.log(f"acc/{prefix}",  outputs['accuracy'])
        self.log(f"ece/{prefix}", outputs['ece'])

    def ensemble_outputs(self, images):
        all_logits = []
        self.models.eval()
        with torch.no_grad():
            for m_idx, m in enumerate(self.models):
                logits = m(images)
                all_logits.append(logits)
        all_logits = torch.stack(all_logits, dim=0)  # num_models x batch_size x num_classes

        # pass through a softmax
        softmax = torch.nn.Softmax(dim=2)
        all_probs = softmax(all_logits)
        # average probs across models
        all_probs = torch.mean(all_probs, dim=0)  # batch x num_classes
        return all_probs
    def predict_step(self, batch):
        all_probs = self.ensemble_outputs(batch[0])
        ts_logits = self.temperature_scale(all_probs)
        return ts_logits, batch[1]

    def forward(self, batch):
        images, labels = batch
        all_probs = self.ensemble_outputs(images)
        ts_logits = self.temperature_scale(all_probs)
        # learn one temperature for all models
        loss = self.criterion(ts_logits, labels)
        accuracy = self.accuracy(ts_logits, labels)
        ece = self.ece_criterion(ts_logits, labels)

        outputs = {'loss': loss,
                   'accuracy': accuracy,
                   'ece': ece}

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        self._call_to_log_outputs(outputs,"train")
        return outputs['loss']

    def configure_optimizers(self):
        if self.optimizer_params is None:
            optimizer_params = {'lr': 0.01,
                                'max_iter': 50,
                                'line_search_fn': 'strong_wolfe'}
        else:
            optimizer_params = self.optimizer_params
        optimizer = optim.LBFGS([self.temperature], **optimizer_params)
        return optimizer


class LogitEnsembleTemperatureScalingTask(ProbEnsembleTemperatureScalingTask):
    def __init__(self, *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

    def ensemble_outputs(self, images):
        all_logits = []
        self.models.eval()
        with torch.no_grad():
            for m_idx, m in enumerate(self.models):
                logits = m(images)
                all_logits.append(logits)
        all_logits = torch.stack(all_logits, dim=0)  # num_models x batch_size x num_classes

        # take the mean
        all_logits = torch.mean(all_logits, dim=0)  # batch x num_classes
        return all_logits

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        return logits / self.temperature



class TemperatureScalingVectorTask(TemperatureScalingTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = nn.Parameter(torch.ones(self.num_classes) * self.init_temperature)

all_ts_modules = {
    "ts_base": TemperatureScalingTask,
    "ts_vector": TemperatureScalingVectorTask,
    "ts_ens_prob" : ProbEnsembleTemperatureScalingTask,
    "ts_ens_log": LogitEnsembleTemperatureScalingTask,
}

