from torch import nn
from torch.optim import Optimizer
import numpy as np
import torch
from torch import Tensor
from typing import List, Optional
import math

class MySGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None, hyper_learning_rate=0):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if (hyper_learning_rate != 0):
                    p.data.add_(-hyper_learning_rate, d_p)
                else:
                    p.data.add_(-group['lr'], d_p)
        return loss


class DANEOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, L = 0.1, eta = 1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, L=L, eta = eta)
        super(DANEOptimizer, self).__init__(params, defaults)

    def step(self, server_grads, pre_grads, pre_params, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            for p, server_grad, pre_grad, pre_param in zip(group['params'], server_grads, pre_grads, pre_params):
                if server_grad.grad.data is not None and pre_grad.data is not None:
                    p.data = p.data - group['lr'] * (p.grad.data - (pre_grad.data - group['eta'] * server_grad.grad.data) + 3 * group['L'] * (p.data - pre_param.data) + group['L'] * p.data )
                else:
                    p.data = p.data - group['lr'] * p.grad.data
        return loss

class Neumann(Optimizer):
    """
    Documentation about the algorithm
    """

    def __init__(self, params, lr=1e-3, eps=1e-8, alpha=1e-7, beta=1e-5, gamma=0.9, momentum=1, sgd_steps=5, K=10):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 1 >= momentum:
            raise ValueError("Invalid momentum value: {}".format(eps))

        self.iter = 0
        # self.sgd = SGD(params, lr=lr, momentum=0.9)

        param_count = np.sum([np.prod(p.size()) for p in params])  # got from MNIST-GAN

        defaults = dict(lr=lr, eps=eps, alpha=alpha,
                        beta=beta * param_count, gamma=gamma,
                        sgd_steps=sgd_steps, momentum=momentum, K=K
                        )

        super(Neumann, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.iter += 1

        loss = None
        if closure is not None:  # checkout what's the deal with this. present in multiple pytorch optimizers
            loss = closure()

        for group in self.param_groups:

            sgd_steps = group['sgd_steps']

            alpha = group['alpha']
            beta = group['beta']
            gamma = group['gamma']
            K = group['K']
            momentum = group['momentum']
            mu = momentum * (1 - (1 / (1 + self.iter)))

            if mu >= 0.9:
                mu = 0.9
            elif mu <= 0.5:
                mu = 0.5

            eta = group['lr'] / self.iter  ## update with time ## changed
            # print("here")

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data).float()
                    state['d'] = torch.zeros_like(p.data).float()
                    state['moving_avg'] = p.data

                if self.iter <= sgd_steps:
                    p.data.add_(-group['lr'], grad)
                    # self.sgd.step()
                    continue

                state['step'] += 1

                # Reset neumann iterate
                if self.iter % K == 0:
                    state['m'] = grad.mul(-eta)
                    ## changed                  

                else:
                    ## Compute update d_t
                    diff = p.data.sub(state['moving_avg'])
                    # # print(diff)
                    # diff_norm = p.data.sub(state['moving_avg']).norm()
                    # if np.count_nonzero(diff) and diff_norm > 0:
                    #    state['d'] = grad.add( (( (diff_norm.pow(2)).mul(alpha) ).sub( (diff_norm.pow(-2)).mul(beta) )).mul( diff.div(diff_norm)) )
                    # else:
                    #    state['d'].add_(grad)
                    state['d'] = grad

                    ## Update Neumann iterate
                    (state['m'].mul_(mu)).sub_(state['d'].mul(eta))

                    ## Update Weights
                    p.data.add_((state['m'].mul(mu)).sub(state['d'].mul(eta)))

                    ## Update Moving Average
                    # state['moving_avg'] = p.data.add( (state['moving_avg'].sub(p.data)).mul(gamma) )

                # print(p.data)

        ## changed
        if self.iter % K == 0:
            group['K'] = group['K'] * 2

        # return loss








class SophiaG(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=1e-1, *, maximize: bool = False,
                 capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= rho:
            raise ValueError("Invalid rho parameter at index 1: {}".format(rho))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, rho=rho,
                        weight_decay=weight_decay,
                        maximize=maximize, capturable=capturable)
        super(SophiaG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()

    def get_hessian(self):
        m = []
        h = []

        return

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError('Hero does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                state_steps.append(state['step'])
                hessian.append(state['hessian'])

                if self.defaults['capturable']:
                    bs = torch.ones((1,), dtype=torch.float, device=p.device) * bs

            sophiag(params_with_grad,
                    grads,
                    exp_avgs,
                    hessian,
                    state_steps,
                    bs=bs,
                    beta1=beta1,
                    beta2=beta2,
                    rho=group['rho'],
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    maximize=group['maximize'],
                    capturable=group['capturable'])

        return loss


def sophiag(params: List[Tensor],
            grads: List[Tensor],
            exp_avgs: List[Tensor],
            hessian: List[Tensor],
            state_steps: List[Tensor],
            capturable: bool = False,
            *,
            bs: int,
            beta1: float,
            beta2: float,
            rho: float,
            lr: float,
            weight_decay: float,
            maximize: bool):
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    func = _single_tensor_sophiag

    func(params,
         grads,
         exp_avgs,
         hessian,
         state_steps,
         bs=bs,
         beta1=beta1,
         beta2=beta2,
         rho=rho,
         lr=lr,
         weight_decay=weight_decay,
         maximize=maximize,
         capturable=capturable)


def _single_tensor_sophiag(params: List[Tensor],
                           grads: List[Tensor],
                           exp_avgs: List[Tensor],
                           hessian: List[Tensor],
                           state_steps: List[Tensor],
                           *,
                           bs: int,
                           beta1: float,
                           beta2: float,
                           rho: float,
                           lr: float,
                           weight_decay: float,
                           maximize: bool,
                           capturable: bool):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda and bs.is_cuda

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        if capturable:
            step = step_t
            step_size = lr
            step_size_neg = step_size.neg()

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            #param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
        else:
            step = step_t.item()
            step_size_neg = - lr

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            #param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)

