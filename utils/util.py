import torch
import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np

import matplotlib.pyplot as plt

# WARNING:
# There is no guarantee that it will work or be used on a model. Please do use it with caution unless you make sure everything is working.
use_fp16 = False

if use_fp16:
    from torch.cuda.amp import autocast
else:
    class Autocast(): # This is a dummy autocast class
        def __init__(self):
            pass
        def __enter__(self, *args, **kwargs):
            pass
        def __call__(self, arg=None):
            if arg is None:
                return self
            return arg
        def __exit__(self, *args, **kwargs):
            pass

    autocast = Autocast()

def rename_parallel_state_dict(state_dict):
    count = 0
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            v = state_dict.pop(k)
            renamed = k[7:]
            state_dict[renamed] = v
            count += 1
    if count > 0:
        print("Detected DataParallel: Renamed {} parameters".format(count))
    return count

def load_state_dict(model, state_dict, no_ignore=False):
    own_state = model.state_dict()
    count = 0
    for name, param in state_dict.items():
        if name not in own_state: # ignore
            print("Warning: {} ignored because it does not exist in state_dict".format(name))
            assert not no_ignore, "Ignoring param that does not exist in model's own state dict is not allowed."
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except RuntimeError as e:
            print("Error in copying parameter {}, source shape: {}, destination shape: {}".format(name, param.shape, own_state[name].shape))
            raise e
        count += 1
    if count != len(own_state):
        print("Warning: Model has {} parameters, copied {} from state dict".format(len(own_state), count))
    return count

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.step = None
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

        self.step = None

    def update(self, key, value, n=1):
        if isinstance(value, tuple) and len(value) == 2:
            value, n = value
        if self.writer is not None:
            self.writer.log_metrics({key: value}, step=self.step)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)

    def set_step(self, step):
        self.step = step


class GroupAccuracyTracker:

    def __init__(self, many_shot, medium_shot, few_shot, num_classes, track_marginal=False):
        self.many_shot = many_shot
        self.medium_shot = medium_shot
        self.few_shot = few_shot
        self.num_classes = num_classes

        self._confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
        self._marginal_tracker = MarginalTracker(self.num_classes) if track_marginal else None

    def reset(self):
        self._confusion_matrix.zero_()

        if self._marginal_tracker is not None:
            self._marginal_tracker.reset()

    def update(self, target, output):
        with torch.no_grad():
            for t, p in zip(target.view(-1), output.argmax(dim=-1).view(-1)):
                self._confusion_matrix[t.long(), p.long()] += 1

            if self._marginal_tracker is not None:
                self._marginal_tracker.update(output)

    def get_confusion_matrix(self, normalize=False):
        cm = self._confusion_matrix

        if normalize:
            cm = cm / cm.sum(1)

        return cm.numpy()

    def accuracy_per_group(self):
        acc = np.diag(self.get_confusion_matrix(normalize=True))

        output = dict(
            many_shot_acc = acc[self.many_shot].mean(),
            medium_shot_acc = acc[self.medium_shot].mean(),
            few_shot_acc = acc[self.few_shot].mean(),
            many_class_num = self.many_shot.sum(),
            medium_class_num = self.medium_shot.sum(),
            few_class_num = self.few_shot.sum(),
        )

        return output

    def accuracy_per_class(self, as_dict=True):
        acc_per_class = np.diag(self.get_confusion_matrix(normalize=True))
        if as_dict:
            return {f'class_{i}_acc': acc_per_class[i] for i in range(len(acc_per_class))}
        else:
            return acc_per_class

    def accuracy(self, balanced=False):
        if balanced:
            acc_per_class = np.diag(self.get_confusion_matrix(normalize=True))
            return acc_per_class.mean()
        else:
            cm = self.get_confusion_matrix(normalize=False)
            return np.diag(cm).sum() / cm.sum()

    def get_marginal_likelihood(self):
        if self._marginal_tracker is None:
            raise AssertionError('Initialize with track_marginal=True to estimate marginal.')

        return self._marginal_tracker.marginal


class MarginalTracker:
    def __init__(self, num_classes):
        self.num_classes = num_classes

        self._marginal_likeli = torch.zeros(self.num_classes)
        self._count = 0

    def update(self, logits_or_probs, normalize=False):
        # probs shape: Batch size x Num classes
        if normalize:
            probs = logits_or_probs.softmax(dim=-1)
        else:
            probs = logits_or_probs

        batch_size = probs.shape[1]
        self._count += batch_size

        # running average
        alpha = (self._count - batch_size) / self._count
        batch_marginal = probs.mean(dim=0).detach()
        self._marginal_likeli = self._marginal_likeli * alpha + batch_marginal * (1 - alpha)

    @property
    def marginal(self):
        return self._marginal_likeli

    def reset(self):
        self._marginal_likeli.zeros_()
        self._count = 0


def plot_confusion_matrix(cm, cmap='jet', verbose=False, title='Confusion matrix', normalized=False):
    num_classes = cm.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(title=title,
           ylabel='True label',
           xlabel='Predicted label')

    if verbose:
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalized else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return fig


def adjusted_model_wrapper(model_orig, test_bias=None):
    def model_fn(x, **kwargs):
        result = model_orig(x, **kwargs)
        if isinstance(result, dict):
            logits = result["logits"]
            ensemble = True
        else:
            logits = result
            ensemble = False

        if test_bias is not None:
            logits = logits + test_bias

        if ensemble:
            final_logits = logits.mean(dim=1)
        else:
            final_logits = logits

        return final_logits

    return model_fn


def setup_wandb_logger(cfg):
    wandb_experiment = os.path.basename(cfg['trainer']['save_dir'])
    wandb_logger = WandbLogger(wandb_experiment, save_dir=cfg.save_dir)
    return wandb_logger


def parse_tau_list(config):
    num_experts_ = config['arch']['args']['num_experts']
    assert num_experts_ >= 1

    try:
        tau_list = config['loss']['args']['tau_list']
    except KeyError:
        print('tau_list is not present. Setting tau_list=None.')
        tau_list = None
    else:
        if isinstance(tau_list, str):
            if tau_list == 'limits':
                tau_min, tau_max = config['loss']['tau_min'], config['loss']['tau_max']
                tau_list = limits_sequence(num_experts_, tau_min, tau_max)

                if num_experts_ == 1:
                    print("To set limit values the minimum number of experts is 2. Setting tau={}".format(tau_list[0]))
            elif tau_list == 'uniform':
                tau_min, tau_max = config['loss']['tau_min'], config['loss']['tau_max']
                tau_list = [tau_min + (tau_max - tau_min) * (i + 0.5) / num_experts_ for i in range(num_experts_)]
            else:
                tau_list = [float(t) for t in tau_list.split(';')]

            assert num_experts_ == len(tau_list)

    return tau_list


def limits_sequence(num_experts_, tau_min, tau_max):
    if num_experts_ == 1:
        # average value
        return [(tau_min + tau_max) / 2.0]
    else:
        # equidistant values in [tau_min ... tau_max]
        return [
            tau_min + (tau_max - tau_min) * i / max(num_experts_ - 1, 1)
            for i in range(num_experts_)
        ]


def seed_everything(seed=1):
    # fix random seeds for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def learning_rate_scheduler(optimizer, config):
    if "type" in config._config["lr_scheduler"]:
        if config["lr_scheduler"]["type"] == "CustomLR": # linear learning rate decay
            lr_scheduler_args = config["lr_scheduler"]["args"]
            gamma = lr_scheduler_args["gamma"] if "gamma" in lr_scheduler_args else 0.1
            print("Scheduler step1, step2, warmup_epoch, gamma:", (lr_scheduler_args["step1"], lr_scheduler_args["step2"], lr_scheduler_args["warmup_epoch"], gamma))
            def lr_lambda(epoch):
                if epoch >= lr_scheduler_args["step2"]:
                    lr = gamma * gamma
                elif epoch >= lr_scheduler_args["step1"]:
                    lr = gamma
                else:
                    lr = 1

                """Warmup"""
                warmup_epoch = lr_scheduler_args["warmup_epoch"]
                if epoch < warmup_epoch:
                    lr = lr * float(1 + epoch) / warmup_epoch
                return lr
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)  # cosine learning rate decay
    else:
        lr_scheduler = None
    return lr_scheduler
