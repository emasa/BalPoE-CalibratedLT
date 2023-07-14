import os

import numpy as np
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
from utils import inf_loop, MetricTracker, load_state_dict, \
    rename_parallel_state_dict, autocast, use_fp16, GroupAccuracyTracker, \
    plot_confusion_matrix, setup_wandb_logger
import model.model as module_arch

from pytorch_lightning.loggers import WandbLogger


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, combiner,
                 finetuning_combiner=None, valid_data_loader=None, val_criterion=None,
                 lr_scheduler=None, len_epoch=None, save_imgs=False):
        super().__init__(model, criterion, metric_ftns, optimizer, config, val_criterion=val_criterion)
        self.config = config

        # add_extra_info will return info about individual experts.
        # This is crucial for individual loss. If this is false, we can only get a final mean logits.
        self.add_extra_info = config._config.get('add_extra_info', False)
        print("self.add_extra_info", self.add_extra_info)

        self.combiner = combiner
        self.finetuning_combiner = finetuning_combiner

        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        if use_fp16:
            self.logger.warn("FP16 is enabled. This option should be used with caution unless you make sure it's working and we do not provide guarantee.")
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.config['trainer'].get('validate', False) and (self.valid_data_loader is not None)
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.save_imgs = save_imgs

        # setup visualization writer instance
        if self.config['trainer'].get('wandb', False):
            self.writer = setup_wandb_logger(self.config)
            self.writer.log_hyperparams(self.config.config)
        else:
            self.writer = None

        train_metrics = ['loss'] + [m.__name__ for m in self.metric_ftns]

        num_experts = config['arch']['args'].get('num_experts', 0)
        return_expert_losses = self.config['loss'].get('return_expert_losses', False)
        if return_expert_losses:
            train_metrics.extend([f'loss_e_{i}' for i in range(num_experts)])

        self.train_metrics = MetricTracker(*train_metrics, writer=self.writer)

        if self.do_validation:
            train_cls_num_list = np.array(data_loader.cls_num_list)
            many_shot = train_cls_num_list > 100
            medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
            few_shot = train_cls_num_list < 20

            val_metrics = ['loss'] + [m.__name__ for m in self.metric_ftns]
            self.valid_metrics = MetricTracker(*val_metrics)
            self.valid_group_acc = GroupAccuracyTracker(many_shot, medium_shot, few_shot, len(train_cls_num_list))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.real_model._hook_before_iter()

        self.train_metrics.reset()
        if self.do_validation:
            self.valid_metrics.reset()

        train_step = None

        if hasattr(self.criterion, "_hook_before_epoch"):
            self.criterion._hook_before_epoch(epoch)

        current_combiner = self._get_combiner(epoch)
        current_combiner.update(epoch)

        for batch_idx, data in enumerate(self.data_loader):
            data, target = data
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                result, loss_dict, acc = current_combiner.forward(
                    self.model, self.criterion, data, target,
                )

            loss = loss_dict['loss']
            if not use_fp16:
                loss.backward()
                self.optimizer.step()
            else:
                self.scaler.s(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            # update global step (ongoing number of batches)
            train_step = (epoch - 1) * self.len_epoch + batch_idx

            self.train_metrics.set_step(train_step)
            for loss_name, loss_value in loss_dict.items():
                self.train_metrics.update(loss_name, loss_value.item())

            output = result['output'] if isinstance(result, dict) else result

            for met in self.metric_ftns:
                if met.__name__ == 'accuracy':
                    self.train_metrics.update(met.__name__, (acc, len(target)))
                else:
                    self.train_metrics.update(met.__name__, met(output, target, return_length=True))

            if batch_idx % self.log_step == 0:
                max_group_lr = max([param_group['lr'] for param_group in self.optimizer.param_groups])
                min_group_lr = min([param_group['lr'] for param_group in self.optimizer.param_groups])
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} max group LR: {:.4f} min group LR: {:.4f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    max_group_lr,
                    min_group_lr
                ))

                if self.writer is not None:
                    if self.save_imgs:
                        self.writer.log_image('train_input', make_grid(data.cpu(), nrow=8, normalize=True), step=train_step)

                    self.writer.log_metrics(dict(max_group_lr=max_group_lr, min_group_lr=min_group_lr), step=train_step)

            if batch_idx == self.len_epoch:
                break

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        log = self.train_metrics.result()
        if self.writer is not None:
            self.writer.log_metrics(log, step=train_step)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            val_log = {f'val_{k}' : v for k, v in val_log.items()}
            log.update(**val_log)
            if self.writer is not None:
                self.writer.log_metrics(val_log, step=epoch)

            # print some results for debugging
            self.logger.info("Accuracy per class")
            acc_per_class = np.diag(self.valid_group_acc.get_confusion_matrix(normalize=True))
            acc_per_class = np.round(acc_per_class * 100, decimals=2)
            with np.printoptions(threshold=np.inf):
                self.logger.info(acc_per_class)

            if epoch == self.epochs:
                self.logger.info("Confusion matrix")
                # log normalized CM data
                cm = self.valid_group_acc.get_confusion_matrix(normalize=True)
                cm = np.round(cm * 100, decimals=2)
                with np.printoptions(threshold=np.inf):
                    self.logger.info(cm)

                if self.writer is not None:
                    # save normalized CM figure
                    self.writer.log_metrics(dict(val_confusion_matrix=plot_confusion_matrix(cm, normalized=True)))

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        self.valid_group_acc.reset()

        with torch.no_grad():
           
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                if isinstance(output, dict):
                    output = output["output"]

                loss = self.val_criterion(output, target)
                self.valid_metrics.update('loss', loss.item())

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, return_length=True))
                self.valid_group_acc.update(target.cpu(), output.cpu())

        log = self.valid_metrics.result()
        per_group_metrics = self.valid_group_acc.accuracy_per_group()
        log['balanced_acc'] = self.valid_group_acc.accuracy(balanced=True)
        log.update(per_group_metrics)

        return log

    def _get_combiner(self, epoch):
        # optional: during the last epochs, train with default combiner (ERM)
        if self.finetuning_combiner is not None and epoch >= self.config['finetuning_combiner']['initial_epoch']:
            return self.finetuning_combiner
        else:
            return self.combiner

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
