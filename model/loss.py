import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
eps = 1e-7 

class BSExpertLoss(nn.Module):
    r"""
    References:

    """

    def __init__(self, cls_num_list=None, tau_list=(0, 1, 2), eps=1e-9, **kwargs):
        super().__init__()
        self.base_loss = F.cross_entropy

        self.register_buffer('bsce_weight', torch.tensor(cls_num_list).float())
        self.register_buffer('tau_list', torch.tensor(tau_list).float())
        self.num_experts = len(tau_list)
        self.eps = eps

        assert self.num_experts >= 1

    def forward(self, output_logits, targets, extra_info=None, return_expert_losses=False):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        if extra_info is None:
            return self.base_loss(output_logits, targets)  # output_logits indicates the final prediction

        logits = extra_info['logits']
        assert len(logits.shape) == 3
        assert logits.shape[0] == self.num_experts

        expert_losses = dict()
        loss = 0.0

        for idx in range(self.num_experts):
            adjusted_expert_logits = logits[idx] + self.get_bias_from_index(idx)
            expert_losses[f'loss_e_{idx}'] = expert_loss = self.base_loss(adjusted_expert_logits, targets)
            loss = loss + expert_loss

        loss = loss / self.num_experts

        if return_expert_losses:
            return loss, expert_losses
        else:
            return loss

    def get_default_bias(self, tau=1):
        prior = self.bsce_weight
        prior = prior / prior.sum()
        log_prior = torch.log(prior + self.eps)
        return tau * log_prior

    def get_bias_from_index(self, e_idx):
        tau = self.tau_list[e_idx]
        return self.get_default_bias(tau)


class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_list=None, reweight_CE=False, **kwargs):
        super().__init__()
        if reweight_CE:
            idx = 1  # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)

        return self

    def forward(self, output_logits, target, **kwargs):  # output is logits
        return F.cross_entropy(output_logits, target, weight=self.per_cls_weights)


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, cls_num_list=None, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
    
    def _hook_before_epoch(self, epoch):
        pass

    def forward(self, output_logits, target):
        return focal_loss(F.cross_entropy(output_logits, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                # CB loss
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)  # * class number
                # the effect of per_cls_weights / np.sum(per_cls_weights) can be described in the learning rate so the math formulation keeps the same.
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)  # one-hot index
         
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1)) 
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s 

        final_output = torch.where(index, x_m, x) 
        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)
        
        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)


class LADELoss(nn.Module):
    def __init__(self, cls_num_list=None, remine_lambda=0.01, **kwargs):
        super().__init__()
        num_classes = len(cls_num_list)
        img_num_per_cls = torch.tensor(cls_num_list).float()

        self.register_buffer('prior', img_num_per_cls / img_num_per_cls.sum())
        self.register_buffer('balanced_prior', torch.tensor(1. / num_classes).float())
        self.register_buffer('cls_weight', img_num_per_cls.float() / torch.sum(img_num_per_cls.float()))
        self.num_classes = num_classes
        self.remine_lambda = remine_lambda  # for cifar-lt: 0.01, for imagenet-lt: 0.5, for inaturalist: 0.1

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)

        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, y_pred, target, q_pred=None):
        """
        y_pred: N x C
        target: N
        """
        per_cls_pred_spread = y_pred.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (y_pred - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)

        loss = -torch.sum(estim_loss * self.cls_weight)
        return loss


class RIDELoss(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True, reweight_epoch=-1, 
        base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05, **kwargs):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0
            
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)   # 这个是logits时算CE loss的weight
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)  # class number
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor   #Eq.3

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)    # the effect can be described in the learning rate so the math formulation keeps the same.

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).cuda()  # 这个是logits时算diversity loss的weight

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None
        else:
            self.per_cls_weights_base = None
            self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0

        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:  
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)
            
            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature
            
            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)
            
            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')
        
        return loss
  
 
class TADELoss(nn.Module):
    def __init__(self, cls_num_list=None, s=30, tau=2, **kwargs):
        super().__init__()
        self.base_loss = F.cross_entropy 
     
        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = tau 

    def reverse_prior(self, prior):
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1  # reverse the order
        backward_prior = value.index_select(0,idx2)
        
        return backward_prior

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction

        loss = 0

        # Obtain logits from each expert  
        expert1_logits = extra_info['logits'][0]
        expert2_logits = extra_info['logits'][1] 
        expert3_logits = extra_info['logits'][2]  
 
        # Softmax loss for expert 1
        loss += self.base_loss(expert1_logits, target)
        
        # Balanced Softmax loss for expert 2 
        expert2_logits = expert2_logits + torch.log(self.prior + 1e-9) 
        loss += self.base_loss(expert2_logits, target)
        
        # Backward Softmax loss for expert 3
        reverse_prior = self.reverse_prior(self.prior)
        expert3_logits = expert3_logits + torch.log(self.prior + 1e-9) - self.tau * torch.log(reverse_prior + 1e-9)
        loss += self.base_loss(expert3_logits, target)
   
        return loss
    
