import torch
import os

from algorithms.edges.edgeSeOrder import edgeSeOrder
from algorithms.edges.edgeFiOrder import edgeFiOrder
from algorithms.edges.edgeDANE import edgeDANE

from algorithms.server.serverbase import ServerBase
from utils.model_utils import read_data, read_edge_data
import numpy as np

# Implementation for FedAvg Server
class Server(ServerBase):
    def __init__(self, dataset,algorithm, model, batch_size, learning_rate, eta, L, num_glob_iters,
                 local_epochs, optimizer, num_edges, times):
        super().__init__(dataset,algorithm, model[0], batch_size, learning_rate, eta, L, num_glob_iters,
                         local_epochs, optimizer, num_edges, times)

        # Initialize data for all  edges
        data = read_data(dataset)
        total_edges = len(data[0])

        for i in range(total_edges):
            id, train, test = read_edge_data(i, data, dataset)
            if(algorithm == "SecondOrder"):
                edge = edgeSeOrder(id, train, test, model, batch_size, learning_rate, eta, L, local_epochs, optimizer)

            if(algorithm == "FirstOrder"):
                edge = edgeFiOrder(id, train, test, model, batch_size, learning_rate, eta, L, local_epochs, optimizer)

            if(algorithm == "DANE"):
                edge = edgeDANE(id, train, test, model, batch_size, learning_rate, eta, L, local_epochs, optimizer)
                
            self.edges.append(edge)
            self.total_train_samples += edge.train_samples
            
        print("Number of edges / total edges:", num_edges, " / ", total_edges)
        print("Finished creating FedNeumann server.")

    def send_grads(self):
        assert (self.edges is not None and len(self.usedgesers) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for edge in self.edges:
            edge.set_grads(grads)

    def train(self):
        loss = []
        if(self.algorithm == "FirstOrder"):
            # All edge will eun GD or SGD to obtain w*
            for edge in self.edges:
                    edge.train(self.local_epochs)
            
            # Communication round
            for glob_iter in range(self.num_glob_iters):
                self.send_parameters()
                self.evaluate()
                self.selected_edges = self.select_edges(glob_iter, self.num_edges)
                for edge in self.selected_edges:
                    print("Update parameter")
                    #edge.update_parameter(self.local_epochs)
                self.aggregate_parameters()

        else: # For DANE and Second Oerder method
            for glob_iter in range(self.num_glob_iters):
                print("-------------Round number: ",glob_iter, " -------------")
                self.send_parameters()
                self.evaluate()
                self.selected_edges = self.select_edges(glob_iter, self.num_edges)

                for edge in self.selected_edges:
                    edge.train(self.local_epochs)

                self.aggregate_parameters()
        self.save_results()
        self.save_model()