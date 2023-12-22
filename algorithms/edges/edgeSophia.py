import torch
import torch.nn as nn
from algorithms.edges.edgebase import Edgebase
from algorithms.optimizers.optimizer import *

class edgeSophia(Edgebase):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, betas, rho,
                 weight_decay, L, local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, betas, rho,
                 weight_decay, L, local_epochs)

        self.pre_params = []
        if (model[1] == "linear_regression"):
            self.loss = nn.MSELoss()
        elif model[1] == "logistic_regression":
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = sophiag(self.model.parameters(), lr=learning_rate, betas=betas, rho=rho,
                 weight_decay=weight_decay, maximize=False, capturable= False)
        # Keep track of local hessians and exp_avg
        self.m = []
        self.h = []
        for group in self.optimizer.param_groups:
            for param in group['params']:
                self.m.append(torch.zeros_like(param, memory_format=torch.preserve_format))
                self.h.append(torch.zeros_like(param, memory_format=torch.preserve_format))



    def train(self, epochs, glob_iter):
        self.model.train()
        # Only update once time
        for X, y in self.trainloaderfull:
            X, y = X.to(self.device), y.to(self.device)
            self.model.train()
            #loss_per_epoch = 0
            # Sample a mini-batch (D_i)
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
