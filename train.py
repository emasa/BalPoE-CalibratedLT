import argparse
import collections
import pprint
import os
import random

import torch
import numpy as np

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import utils.combiner as module_combiner
from parse_config import ConfigParser
from trainer import Trainer
from utils import write_json, parse_tau_list, seed_everything, learning_rate_scheduler


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_class = getattr(module_loss, config["loss"]["type"])
    extra_parameters = {}

    if hasattr(loss_class, "require_num_experts") and loss_class.require_num_experts:
        extra_parameters["num_experts"] = config["arch"]["args"]["num_experts"]

    criterion = config.init_obj('loss', module_loss, cls_num_list=data_loader.cls_num_list, **extra_parameters)

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler.
    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
    lr_scheduler = learning_rate_scheduler(optimizer, config)

    combiner = config.init_obj('combiner', module_combiner, cfg=config)

    finetuning_combiner = None
    try:
        initial_epoch = config['finetuning_combiner']['initial_epoch']
    except KeyError:
        print('Finetuning disabled.')
    else:
        if config['trainer']['epochs'] > initial_epoch >= 0:
            print('Finetuning starts at epoch:', initial_epoch)
            finetuning_combiner = config.init_obj('finetuning_combiner', module_combiner, cfg=config)
        else:
            print('Finetuning disabled.')

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config, data_loader, combiner,
                      finetuning_combiner=finetuning_combiner,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-l', '--log-config', default='logger/logger_config.json', type=str,
                      help='logging config file path (default: logger/logger_config.json)')
    args.add_argument('-s', '--seed', default=1, type=int,
                      help='random seed (default: 1)')
    args.add_argument("--val", "--validate", dest="validate", action="store_true", default=False,
                      help="Run validation (default: False).")

    args.add_argument("--use-wandb", dest="use_wandb", action="store_true", default=False,
                      help="Use wandb logger (Requires wandb installed and an API key)")

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--n_gpu'], type=int, target='n_gpu'),
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--wd', '--weight_decay'], type=float, target='optimizer;args;weight_decay'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--data_dir'], type=str, target='data_loader;args;data_dir'),
        CustomArgs(['--name'], type=str, target='name'),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--step1'], type=int, target='lr_scheduler;args;step1'),
        CustomArgs(['--step2'], type=int, target='lr_scheduler;args;step2'),
        CustomArgs(['--warmup'], type=int, target='lr_scheduler;args;warmup_epoch'),
        CustomArgs(['--gamma'], type=float, target='lr_scheduler;args;gamma'),
        CustomArgs(['--t_max'], type=int, target='lr_scheduler;args;T_max'),
        CustomArgs(['--eta_min'], type=float, target='lr_scheduler;args;eta_min'),
        CustomArgs(['--save_period'], type=int, target='trainer;save_period'),
        CustomArgs(['--save_dir'], type=str, target='trainer;save_dir'),
        CustomArgs(['--reduce_dimension'], type=int, target='arch;args;reduce_dimension'),
        CustomArgs(['--layer2_dimension'], type=int, target='arch;args;layer2_output_dim'),
        CustomArgs(['--layer3_dimension'], type=int, target='arch;args;layer3_output_dim'),
        CustomArgs(['--layer4_dimension'], type=int, target='arch;args;layer4_output_dim'),
        CustomArgs(['--num_experts'], type=int, target='arch;args;num_experts'),
        CustomArgs(['--use_norm'], type=int, target='arch;args;use_norm'),
        CustomArgs(['--norm_scale'], type=float, target='arch;args;s'),
        CustomArgs(['--tau_list'], type=str, target='loss;args;tau_list'),
        CustomArgs(['--tau_max'], type=float, target='loss;tau_max'),
        CustomArgs(['--tau_min'], type=float, target='loss;tau_min'),
        CustomArgs(['--combiner_type'], type=str, target='combiner;mode'),
        CustomArgs(['--mixup_alpha'], type=float, target='combiner;mixup;alpha'),
        CustomArgs(['--cutmix_alpha'], type=float, target='combiner;mixup;cutmix_alpha'),
        CustomArgs(['--target_mix_strategy'], type=str, target='combiner;mixup;target_mix_strategy'),
        CustomArgs(['--randaugm'], type=int, target='data_loader;args;randaugm'),
        CustomArgs(['--cutout'], type=int, target='data_loader;args;cutout'),
        CustomArgs(['--trivialaugm'], type=int, target='data_loader;args;trivialaugm'),
        CustomArgs(['--finetuning_start'], type=int, target='finetuning_combiner;initial_epoch'),
        CustomArgs(['--track_expert_losses'], type=int, target='loss;return_expert_losses'),
        CustomArgs(['--share_layer2'], type=int, target='arch;args;share_layer2'),
        CustomArgs(['--share_layer3'], type=int, target='arch;args;share_layer3'),
    ]

    config, args = ConfigParser.from_args(args, options)

    num_experts = config['arch']['args'].get('num_experts', None)
    if num_experts is None or num_experts == 1:
        # returns_feat is not supported for a single model
        config['arch']['args'].pop('returns_feat', None)

    # after parsing config, override tau_list if necessary
    config['loss']['args']['tau_list'] = parse_tau_list(config)

    write_json(config.config, config.save_dir / 'config.json')

    pprint.pprint(config)

    deterministic = True
    if deterministic:
        seed_everything(args.seed)

    main(config)
