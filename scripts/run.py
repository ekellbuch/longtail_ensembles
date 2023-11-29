#!/usr/bin/env python
"""
Train a model on CIFAR10 or CIFAR100

"""
from omegaconf import DictConfig, OmegaConf, open_dict
import os
import shutil

import hydra
from tqdm import tqdm
import datetime
import torch
import json
import numpy as np
from omegaconf import OmegaConf

from pytorch_lightning.callbacks import ProgressBar
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from longtail_ensembles.callback import GradNormCallbackSplit
from longtail_ensembles.module import all_modules
from longtail_ensembles.data import all_datasets
from pytorch_data.utils import count_classes

from longtail_ensembles.temperature_scaling import all_ts_modules, ModuleWithTemperature
import copy
import wandb

script_dir = os.path.abspath(os.path.dirname(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def store_models(model, ind_data,ood_data, args, logit_dir, suffix=''):

    # store individual model logits
    store_split = args.eval_cfg.get('store_split', False)
    # apply softmax before storing logits
    softmax = bool(args.eval_cfg.get('softmax', False))

    preds_ind, labels_ind = custom_eval(
        model,
        ind_data,
        device,
        softmax=softmax,
        store_split=store_split)

    preds_ood, labels_ood = custom_eval(
        model,
        ood_data,
        device,
        softmax=softmax,
        store_split=store_split)

    if args.eval_cfg.get("store_train", False):
        preds_train, labels_train = traindata_eval(model,
                                                   ind_data,
                                                   device,
                                                   softmax=softmax,
                                                   store_split=store_split)

        np.save("train_preds{}".format(suffix), preds_train)
        np.save("train_labels{}".format(suffix), labels_train)

    if not os.path.exists(logit_dir):
        os.makedirs(logit_dir)
    os.chdir(logit_dir)
    full_path = "."

    # Store model logits for ind data
    np.save("ind_preds{}".format(suffix), preds_ind)
    np.save("ind_labels{}".format(suffix), labels_ind)

    # store model logits for ood data
    if args.data_cfg.ood_dataset == "cifar10_1":
        np.save("ood_preds{}".format(suffix), preds_ood)
        np.save("ood_labels{}".format(suffix), labels_ood)
    else:
        name = "ood_{}".format(args.data_cfg.ood_dataset)
        if "_c" in args.data_cfg.ood_dataset:
            name += "_{}_{}".format(args.data_cfg.corruption, args.data_cfg.level)

        np.save("{}_preds{}.npy".format(name, suffix), preds_ood)
        np.save("{}_labels{}.npy".format(name, suffix), labels_ood)
    return

def traindata_eval(model, ind_data, device, softmax=True, store_split=True):
    """Custom evaluation function to output logits as arrays from models given the trained model on the training data. Used to generate training examples from random labels.

    :param model: a model from interpensembles.modules. Should have a method "forward_outs" that outputs predictions (logits) and labels given images and labels.
    :param ind_data: an instance of a data class (like CIFAR10Data,CIFAR10_1Data) that has a corresponding test_dataloader.
    :param device: device to run computations on.
    :param softmax: whether or not to apply softmax to predictions.
    :returns: four arrays corresponding to predictions (array of shape (batch,classes)), and labels (shape (batch,)) for ind and ood data respectively.

    """
    model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(
                enumerate(ind_data.train_dataloader(shuffle=False,
                                                    aug=False))):
            ims = batch[0].to(device)
            labels = batch[1].to(device)
            pred, label = model.forward_outs((ims, labels),
                                            store_split=store_split,
                                            use_softmax=softmax,
                                            )
            # to cpu
            predarray = pred.cpu().numpy()  # 256x10
            labelarray = label.cpu().numpy()  #
            if idx == 0:
                all_preds_array = predarray
                all_labels_array = labelarray
            else:
                all_preds_array = np.concatenate((all_preds_array, predarray), axis=0)
                all_labels_array = np.concatenate((all_labels_array, labelarray), axis=0)
    return all_preds_array, all_labels_array


def custom_eval(model,
                ind_data,
                device,
                softmax=True,
                store_split=False):
    """Custom evaluation function to output logits as arrays from models given the trained model, in distribution data and out of distribution data.

    :param model: a model from interpensembles.modules. Should have a method "forward_outs" that outputs predictions (logits) and labels given images and labels.
    :param ind_data: an instance of a data class (like CIFAR10Data,CIFAR10_1Data) that has a corresponding test_dataloader.
    :param device: device to run computations on.
    :param softmax: whether or not to apply softmax to predictions.
    :returns: four arrays corresponding to predictions (array of shape (batch,classes)), and labels (shape (batch,)) for ind and ood data respectively.

    """
    # This is the only place where we need to worry about devices
    model.to(device)
    model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(ind_data.test_dataloader())):
            ims = batch[0].to(device)
            labels = batch[1].to(device)
            pred, label = model.forward_outs((ims, labels),
                                            store_split=store_split,
                                            use_softmax=softmax)
            # to cpu
            predarray = pred.cpu().numpy()  # 256x10
            labelarray = label.cpu().numpy()  #
            if idx == 0:
                all_preds_ind_array = predarray
                all_labels_ind_array = labelarray
            else:
                all_preds_ind_array = np.concatenate((all_preds_ind_array, predarray), axis=0)
                all_labels_ind_array = np.concatenate((all_labels_ind_array, labelarray), axis=0)

    return all_preds_ind_array, all_labels_ind_array


@hydra.main(config_path="script_configs", config_name="run_gpu_cifar10", version_base=None)
def main(args):
    train(args)
    return


def train(args):
    #%%
    if args.seed is not None:
        seed_everything(args.seed)

    try:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        output_dir = hydra_cfg['runtime']['output_dir']
    except:
        pass

    # Set up logging.
    project_name = args.get('project_name', args.data_cfg.test_set)
    if args.trainer_cfg.logger == "wandb":
        if args.trainer_cfg.accelerator == "ddp":
            kwargs = {"group":"DDP"}
        else:
            kwargs = dict()

        logger = WandbLogger(name=args.module_cfg.classifier,
                             project=project_name, **kwargs)
    elif args.trainer_cfg.logger == "tensorboard":
        logger = TensorBoardLogger(project_name,
                                   name=args.module_cfg.classifier,
                                   )
    else:
        logger = None

    if args.trainer_cfg.logger == "wandb":
        # multi gpu compat
        # https://github.com/Lightning-AI/lightning/issues/5319#issuecomment-869109468
        if os.environ.get("LOCAL_RANK", None) is None:
            os.environ["EXP_LOG_DIR"] = logger.experiment.dir
        args_as_dict = OmegaConf.to_container(args)
        logger.log_hyperparams(args_as_dict)

    # path where data is saved
    today_str = datetime.datetime.now().strftime("%y-%m-%d")
    ctime_str = datetime.datetime.now().strftime("%H-%M-%S")


    experiment_time_str = today_str + "/" + ctime_str
    checkpoint_dir = os.path.join(script_dir,
                                  "../",
                                  "models",
                                  args.module_cfg.classifier,
                                  args.module_cfg.module,
                                  today_str,
                                  ctime_str,
                                  )

    # Configure checkpoint and trainer:
    checkpoint = ModelCheckpoint(
        monitor="accuracy/val",
        mode="max",
        save_last=False,
        dirpath=checkpoint_dir,
    )

    trainerargs = OmegaConf.to_container(args.trainer_cfg)
    trainerargs['logger'] = logger

    if args.trainer_cfg.accelerator == "ddp":
            trainerargs['plugins'] = [
                DDPStrategy(find_unused_parameters=False)
        ]

    all_callbacks = []
    if args.callbacks:
        if args.callbacks.gradnorm:
            all_callbacks.append(GradNormCallbackSplit())
        if args.callbacks.checkpoint_callback:
            all_callbacks.append(checkpoint)
        if args.callbacks.early_stopping:
            all_callbacks.append(EarlyStopping(**args.early_stop_cfg))
        if args.callbacks.get('progress_bar', None):
            all_callbacks.append(ProgressBar(refresh_rate=10))
        if args.callbacks.get('lr_monitor', None):
            all_callbacks.append(LearningRateMonitor(logging_interval='step'))

    trainer = Trainer(**trainerargs, callbacks=all_callbacks)

    # TODO: Effective batch size is split across gpus
    # https://github.com/Lightning-AI/lightning/discussions/3706
    if len(trainer.device_ids) > 1:
        num_gpus = len(trainer.device_ids)
        args.data_cfg.batch_size = int(args.data_cfg.batch_size / num_gpus)
        args.data_cfg.num_workers = int(args.data_cfg.num_workers / num_gpus)

    # ------------------------------------------------
    # Train set
    ind_data = all_datasets[args.data_cfg.test_set](args.data_cfg)
    ind_data.setup()
    # ------------------------------------------------
    # OOD set
    ood_data = all_datasets[args.data_cfg.ood_dataset](args.data_cfg)
    ood_data.setup()
    # ------------------------------------------------
    # add num_classes and samples_per_class to module_cfg
    module_args = args.module_cfg
    OmegaConf.set_struct(module_args, True)
    with open_dict(module_args):
        module_args.num_classes = args.data_cfg.num_classes

    if "bloss" or "weighted" in module_args.module:
        samples_per_class = count_classes(ind_data, args.data_cfg.num_classes)
        module_args.samples_per_class = samples_per_class.tolist()

    module_args = {"hparams": module_args}

    # we can load in pretrained models stored as weights.
    if bool(args.eval_cfg.test_phase) and not bool(args.module_cfg.pretrained):
        if os.path.isdir(args.module_cfg.checkpoint):
            tmp_files = os.listdir(args.module_cfg.checkpoint)
            assert len(tmp_files) == 1, "more than one file in directory"
            state_dict = os.path.join(args.module_cfg.checkpoint, tmp_files[0])
        else:
            state_dict = args.module_cfg.checkpoint

        ckpt = torch.load(state_dict)
        model = all_modules[args.module_cfg.module](**module_args)
        if 'state_dict_best' in ckpt:
            weights = ckpt['state_dict_best']['feat_model']
            weights.update(ckpt['state_dict_best']['classifier'])
            weights = {k.replace('module.', ''): v for k, v in weights.items()}
            model.model.load_state_dict(weights)
        else:
            model.load_state_dict(ckpt["state_dict"])

    else:  # if training from scratch or loading from state dict:
        model = all_modules[args.module_cfg.module](**module_args)
        #  if loading from state dictionary instead of checkpoint:
        if bool(args.module_cfg.pretrained):
            if args.pretrained_path is None:
                state_dict = os.path.join(script_dir, "../", "models",
                                          "cifar10_models", "state_dicts",
                                          args.module_cfg.classifier + ".pt")
            else:
                if os.path.isdir(args.module_cfg.pretrained_path):
                    tmp_files = os.listdir(args.module_cfg.pretrained_path)
                    assert len(
                        tmp_files) == 1, "more than one file in directory"
                    state_dict = os.path.join(args.module_cfg.pretrained_path, tmp_files[0])
                else:
                    state_dict = args.module_cfg.pretrained_path
            model.model.load_state_dict(torch.load(state_dict))

    # log where checkpoints and logits are stored:
    logit_dir = os.path.join(os.getcwd(), "outputs", experiment_time_str)

    # add output logit dir to args
    if args.trainer_cfg.logger == "wandb":
        try:
            logger.experiment.summary['out/checkpoint_dir'] = checkpoint.dirpath
            logger.experiment.summary['out/logits_dir'] = logit_dir
        except:
            pass

    # do we train the model or not?
    if bool(args.eval_cfg.test_phase):
        # Let's add some metrics for the train set
        # trainer.test(model, ind_data.train_dataloader())[0]
        pass
    else:
        trainer.fit(model, ind_data, **args.fit_cfg)

    # ------------------------------------------------
    # Evaluate model on datasets
    trainer.test(model, ind_data.test_dataloader())

    # Log OOD metrics
    model.logging_stage_name = 'ood/'
    trainer.test(model, ood_data.test_dataloader())

    print('Finished logging ind/ood performance', flush=True)
    # make sure we log the right things:
    if bool(args.eval_cfg.return_model):
        # debug mode
        return ind_data, ood_data, model, trainer

    # store logits
    if not bool(args.eval_cfg.random_eval):
        store_models(model, ind_data, ood_data, args, logit_dir)

    # ------------------------------------------------
    # Let's compare w and wo temperature scaling
    if bool(args.module_cfg.temperature_scaling):
        print('\nStarting temperature scaling run', flush=True)

        if args.ts_cfg.ts_module == "ts_vector":
            samples_per_class = count_classes(ind_data, args.data_cfg.num_classes)
            init_temp = samples_per_class/samples_per_class.max()*args.ts_cfg.init_temp
        else:
            init_temp = args.ts_cfg.init_temp
        ts_model = all_ts_modules[args.ts_cfg.ts_module](model,
                                          init_temp=init_temp,
                                          use_train_loss=args.ts_cfg.use_train_loss,
                                          opt_params=args.ts_cfg.get('opt_params', None),
                                          )
        ts_trainerargs = copy.deepcopy(trainerargs)
        ts_trainerargs['logger'] = None
        ts_trainerargs['max_epochs'] = args.ts_cfg.max_epochs
        ts_trainerargs.update(args.ts_cfg.get('trainer_args', {}))
        # estimate the temperature
        ts_trainer = Trainer(**ts_trainerargs)
        ts_trainer.fit(model=ts_model, train_dataloaders=ind_data.val_dataloader())
        ts_model.temperature.requires_grad = False
        # set temperature
        print('\ntemperature scaling factor: T={}'.format(ts_model.temperature.numpy()), flush=True)
        # if not ensemble
        model.model = ModuleWithTemperature(model.model, ts_model.temperature)
        # if ensemble
        # model.models = ModuleWithTemperature(model.models, ts_model.temperature)

        # Log metrics before/after temperature scaling:
        model.logging_stage_name = 'ind_temperature/'
        trainer.test(model, ind_data.test_dataloader())

        model.logging_stage_name = 'ood_temperature/'
        trainer.test(model, ood_data.test_dataloader())

        # store logits
        if not bool(args.eval_cfg.random_eval):
            store_models(model, ind_data, ood_data, args, logit_dir, suffix='_temperature')


    print("saving metadata to meta.json in directory", flush=True)
    metadata = {}
    metadata["out/checkpoint_dir"] = trainer.checkpoint_callback.dirpath
    with open("meta.json", "w") as f:
        json.dump(metadata, f)

    # dumps to file:
    print("saving config in metadata directory", flush=True)
    with open("config.yaml", "w") as f:
        OmegaConf.save(args, f)

    print("saved metadata to meta.json in directory", trainer.checkpoint_callback.dirpath, flush=True)

    # write metadata
    if args.trainer_cfg.logger == "wandb":
        wandb.finish()

    #%%


if __name__ == "__main__":
    main()
