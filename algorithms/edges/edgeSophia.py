import torch
import torch.nn as nn
from algorithms.edges.edgebase import Edgebase
from algorithms.optimizers.optimizer import *
from torch.functional import F

class edgeSophia(Edgebase):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, alpha, eta, L,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, alpha, eta, L,
                         local_epochs)

        self.pre_params = []
        if (model[1] == "linear_regression"):
            self.loss = nn.MSELoss()
        elif model[1] == "logistic_regression":
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.CrossEntropyLoss()#nn.NLLLoss()
        #print(alpha, eta)
        self.optimizer = SophiaG(self.model.parameters(), lr=learning_rate, betas=alpha, rho=eta,
                                 weight_decay=L, maximize=False, capturable=False)
        # Keep track of local hessians and exp_avg
        self.m = []
        self.h = []
        self.k = 4
        for group in self.optimizer.param_groups:
            for param in group['params']:
                self.m.append(torch.zeros_like(param, memory_format=torch.preserve_format))
                self.h.append(torch.zeros_like(param, memory_format=torch.preserve_format))



    def train(self, epochs, glob_iter):
        # Only update once time
        self.model.train()
        for i, (X, y) in zip(range(1), self.trainloaderfull):
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.step()
            logits = self.model(X)
            loss = self.loss(logits, y)
            loss.backward()
            self.optimizer.step(bs=self.batch_size)
            self.m, self.h = self.optimizer.get_m_h()
            self.optimizer.zero_grad(set_to_none=True)

            if glob_iter % self.k != self.k - 1:
                continue
            else:
                # update hessian EMA
                logits = self.model(X)
                samp_dist = torch.distributions.Categorical(logits=logits)
                y_sample = samp_dist.sample()
                loss_sampled = F.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.view(-1), ignore_index=-1)
                loss_sampled.backward()
                self.optimizer.update_hessian()
                self.optimizer.zero_grad(set_to_none=True)


    def send_grad(self):
        return copy.deepcopy(self.m)


    def send_hessian(self):
        return copy.deepcopy(self.h)
